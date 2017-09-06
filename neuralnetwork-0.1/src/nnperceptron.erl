-module(nnperceptron).

-include("nn_type.hrl").

-define(ETA, 0.3).
-define(BACKPROP_FACTOR, 1).

% a perceptron is a server
-behaviour(gen_server).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2, code_change/3]).

-export([new/2, from_term/1, stop/1, connect/2, to_term/1, to_c/1, link_up/1, get_id/1, pass/2, backprop/2, show/1]).

new(F, ID) ->
    {ok, Pid} = gen_server:start_link(?MODULE, {F, ID}, []),
    Pid.

from_term(Term) ->
    {ok, Pid} = gen_server:start_link(?MODULE, {from_term, Term}, []),
    Pid.

stop(Pid) -> gen_server:call(Pid, stop).

connect(From, To) ->
    gen_server:call(From, {add_output, To}),
    gen_server:call(To, {add_input, From}).

to_term(Pid) ->
    gen_server:call(Pid, to_term).

to_c(Pid) ->
    gen_server:call(Pid, to_c).

link_up(Pid) ->
    gen_server:cast(Pid, link_up).

get_id(Pid) ->
    gen_server:call(Pid, get_id).

pass(Pid, V) ->
    gen_server:cast(Pid, {pass, V}).

backprop(Pid, E) ->
    gen_server:cast(Pid, {backprop, E}).

show(Pid) when is_pid(Pid) ->
    gen_server:call(Pid, show);
show(Id) when is_integer(Id) ->
    show(nnetwork:id2pid(Id)).

% ------------------------------
%  gen-server
% ------------------------------
init({from_term, T}) ->
    ID = proplists:get_value(id, T),
    Ins = proplists:get_value(inputs, T),
    Ous = proplists:get_value(outputs, T),
    State = #perceptron{
                         id = ID,
                         inputs = Ins,
                         outputs = Ous,
                         weights = proplists:get_value(weights, T), 
                         fft = proplists:get_value(fft, T),
                         inputstate = lists:duplicate(length(Ins), undefined),
                         laststate = undefined,
                         sensitivity = lists:duplicate(length(Ous), undefined),
                         total = length(Ins)
                     },
    {ok, State};
init({T, ID}) ->
    rand:seed(exrop),
    State = #perceptron{
        id = ID,
        fft = T
    },
    {ok, State}.

handle_call(get_id, _From, #perceptron{id = ID} = State) ->
    {reply, ID, State};
handle_call({add_output, ToID}, _From, #perceptron{outputs = Outputs, sensitivity = S} = State) ->
    {reply, ok,
        case lists:member(ToID, Outputs) of
            false -> State#perceptron{outputs = [ToID | Outputs], 
                                      sensitivity = [undefined | S]};
            _ -> State
        end};
handle_call({add_input, FromID}, _From, #perceptron{inputs = Inputs, 
             inputstate = IS, weights = WS, total = N} = State) ->
    {reply, ok,
        case lists:member(FromID, Inputs) of
            false -> 
                {ok , W} = (State#perceptron.accept)(FromID),
                State#perceptron{inputs = [FromID | Inputs], 
                                 inputstate = [undefined | IS], 
                                 weights = [W | WS], total = N + 1};
            _ -> State
        end};
handle_call(to_term, _From, #perceptron{id = ID, inputs = Inputs, 
                                     weights = WS, outputs = OS, fft = T} = State) ->
    {reply, [{id, ID}, {inputs, to_ids(Inputs)}, {weights, WS}, {outputs, to_ids(OS)}, {fft, T}], State};
    %{reply, [{id, ID}, {inputs, to_ids(Inputs)}, {weights, WS}, {fft, T}], State};
handle_call(to_c, _From, #perceptron{id = ID, inputs = [], 
                                     weights = WS, outputs = OS, fft = T} = State) ->
    Str = io_lib:format("x(~p)", [ID]),
    {reply, Str, State};
handle_call(to_c, _From, #perceptron{id = ID, inputs = Inputs, 
                                     weights = WS, outputs = OS, fft = T} = State) ->
    Str0 = lists:flatten(lists:zipwith(fun (W, I) -> io_lib:format("~p * (~s) + ", [W,
                                                          nnperceptron:to_c(I)])
                  end,
                  lists:droplast(WS),
                  Inputs)),
    Str = io_lib:format("~s(~s~p)", [atom_to_list(T), Str0, lists:last(WS)]),
    {reply, Str, State};
handle_call(show, _From, #perceptron{id = ID, inputs = Inputs, inputstate = _IS, 
                                     weights = WS, outputs = OS, fft = T, sensitivity = _Se} = State) ->
    io:format("id: ~p~nIn: ~p~nOu: ~p~nWi: ~p~nTy: ~p~n", 
        [{self(), ID}, Inputs, OS, WS, T]),
    {reply, ok, State};
handle_call(stop, _From, _State) ->
    {stop, normal, stopped, #perceptron{}}.

handle_cast({pass, V}, #perceptron{outputs = Outputs} = State) ->
    lists:foreach(fun (O) -> gen_server:cast(O, {stimulate, self(), V}) end, Outputs),
    {noreply, State};

handle_cast({backprop, FromID, V}, 
    #perceptron{inputs = [], outputs = Outputs, sensitivity = IS} = State) ->
    case add_input(FromID, V, Outputs, IS) of
        {_, 0} -> 
            nnetwork:done_backprop(self()),
            {noreply, State#perceptron{sensitivity = lists:duplicate(length(Outputs), undefined)}};
        {NewIS10, _} ->
            {noreply, State#perceptron{sensitivity = NewIS10}}
    end;
handle_cast({backprop, Err}, State) ->
    {noreply, do_backprop(Err, State)};
handle_cast({backprop, FromID, V}, 
    #perceptron{outputs = Outputs,
                sensitivity = IS} = State) ->
    case add_input(FromID, V, Outputs, IS) of
        {NewIS10, 0} -> 
            State10 = do_backprop(lists:sum(NewIS10), State),
            {noreply, State10#perceptron{sensitivity =
                    lists:duplicate(length(Outputs), undefined)}};
        {NewIS10, _} ->
            {noreply, State#perceptron{sensitivity = NewIS10}}
    end;
handle_cast({stimulate, FromID, V}, 
    #perceptron{inputs = Inputs, outputs = Outputs, inputstate = IS, weights = Weights, fft = T, total = N} = State) ->
    case add_input(FromID, V, Inputs, IS) of
        {NewIS10, 0} -> 
            {X, Y} = calc([1 | NewIS10], Weights, T),
            case Outputs of
                [] -> nnetwork:done(self(), Y);
                _ -> lists:foreach(fun (O) -> gen_server:cast(O, {stimulate, self(), Y}) end, Outputs)
            end,
            {noreply, 
                State#perceptron{inputstate = lists:duplicate(N, undefined), 
                                 laststate = NewIS10, 
                                 net = X, last = Y}};
        {NewIS10, _} ->
            {noreply, State#perceptron{inputstate = NewIS10}}
    end;
handle_cast(link_up, 
    #perceptron{inputs = Inputs, outputs = Outputs} = State) ->
    I = lists:map(fun (Id) -> nnetwork:id2pid(Id) end, Inputs),
    O = lists:map(fun (Id) -> nnetwork:id2pid(Id) end, Outputs),
    {noreply, State#perceptron{inputs = I, outputs = O}}.

handle_info(_Msg, State) ->
    {noreply, State}.

terminate(_, _State) ->
    {stop, normal, #perceptron{}}.

code_change(_OldVsn, State, _Extra) -> {ok, State}.

do_backprop(Err, #perceptron{inputs = Inputs, laststate = LS, weights =
        Weights, fft = T} = State) ->
    ErrD = Err * deriv(T, State#perceptron.net, State#perceptron.last),
    backprop_0(ErrD, tl(Weights), Inputs),
    Weights10 = lists:zipwith(fun (W, X) -> ?ETA * ErrD * X + W end,
        Weights, [1 | LS]),
    State#perceptron{weights = Weights10}.

backprop_0(Error, Ws, Ps) ->
    F = ?BACKPROP_FACTOR * Error,
    lists:zipwith(fun (Id, W) -> 
        gen_server:cast(Id, {backprop, self(),  F * W}) end, Ps, Ws).

add_input(ID, V, Inputs, InputState) ->
    {NewState, true, N} = add_input0(ID, V, Inputs, InputState, [], false, 0),
    {NewState, N}.

add_input0(_ID, _V, [], _InputState, AccS, Found, Nundefined) ->
    {lists:reverse(AccS), Found, Nundefined};
add_input0(ID, V, [ID | TI], [undefined | TS], AccS, _Found, Nundefined) ->
    add_input0(ID, V, TI, TS, [V | AccS], true, Nundefined);
add_input0(ID, V, [_ | TI], [undefined | TS], AccS, Found, Nundefined) ->
    add_input0(ID, V, TI, TS, [undefined | AccS], Found, Nundefined + 1);
add_input0(ID, V, [_ | TI], [X | TS], AccS, Found, Nundefined) ->
    add_input0(ID, V, TI, TS, [X | AccS], Found, Nundefined).

to_ids(Pids) -> lists:map(fun (Pid) -> get_id(Pid) end, Pids).

inner_product(V, W) ->
    lists:sum(lists:zipwith(fun (X, Y) -> X * Y end, V, W)).

calc(Input, Weight, F) ->
    X = inner_product(Input, Weight),
    {X, activation(F, X)}.

activation(identity, X) -> X;
activation(binary, X) -> if X >= 0 -> 1; true -> 0 end;
activation(sigmoid, X) -> 1 / (1 + math:exp(-X));
activation(tanh, X) -> math:tanh(X);
activation(arctan, X) -> math:atan(X);
activation(rectified, X) -> if X >= 0 -> X; true -> 0 end;
activation(softplus, X) -> math:log(1 + math:exp(X));
activation(bent, X) -> X + (math:sqrt(X * X + 1) - 1) / 2;
activation(sin, X) -> math:sin(X);
activation(sinc, X) -> if X == 0.0 -> 1; true -> math:sin(X)/X end;
activation(gaussian, X) -> math:exp(-X * X).

deriv(identity, _X, _Y) -> 1;
deriv(binary, _X, _Y) -> 0;
deriv(sigmoid, _X, Y) -> (1 - Y) * Y;
deriv(tanh, _X, Y) -> 1 - Y * Y;
deriv(arctan, X, _Y) -> 1 / (1 + X * X);
deriv(rectified, X, _Y) -> if X >= 0 -> 1; true -> 0 end;
deriv(softplus, X, _Y) -> 1 / (1 + math:exp(-X));
deriv(bent, X, _Y) -> X / 2 / math:sqrt(X * X + 1) + 1;
deriv(sin, X, _Y) -> math:cos(X);
deriv(sinc, X, Y) -> if X == 0.0 -> 0; true -> (math:cos(X) - Y) / X end;
deriv(gaussian, X, Y) -> -2 * X * Y.
