-module(nnetwork).

-export([start/0, start/1, start/2, stop/0, show/0]).

-export([id2pid/1, save/1, load/1, to_term/0, to_c/0, sync_pass/1, pass/1, backprop/1, 
         sync_backprop/1, done/2, done_backprop/1, 
         set_on_done/1, set_on_done_backprop/1]).

-behaviour(gen_server).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2, code_change/3]).

-record(network, 
    {
        id2pid = dict:new(),
        pid2id = dict:new(),

        ni, outlayer, 
        layers = [],
        ondone = fun (V) -> io:format("ANN: ~p~n", [V]) end,
        ondone_backprop,
        backprop_set = sets:new(),
        output_v
    }).

start() -> start({3, [6], 2}).

start({NI, NHs, NO}) ->
    start({NI, NHs, NO}, sigmoid).

start({NI, NHs, NO}, ActF) ->
    {ok, Pid} = gen_server:start_link({local, ?MODULE}, ?MODULE, {{NI, NHs, NO},
            ActF}, []),
    Pid.

stop() ->
    gen_server:call(?MODULE, stop).

show() ->
    gen_server:cast(?MODULE, show).

id2pid(Id) ->
    gen_server:call(?MODULE, {id2pid, Id}).

%pid2id(Pid) ->
%    gen_server:call(?MODULE, {pid2id, Pid}).

set_on_done(F) ->
    gen_server:call(?MODULE, {set_done, F}).

set_on_done_backprop(F) ->
    gen_server:call(?MODULE, {set_done_backprop, F}).

sync_pass(V) ->
    F = set_on_done(self()),    
    pass(V),
    R10 = receive {done, R} -> R end,
    set_on_done(F),
    R10.

pass(V) ->
    gen_server:cast(?MODULE, {pass, V}).

backprop(V) ->
    gen_server:cast(?MODULE, {backprop, V}).

sync_backprop(V) ->
    F = set_on_done_backprop(self()),    
    backprop(V),
    receive done_backprop -> ok after 10000 -> timedout end,
    set_on_done_backprop(F),
    ok.

done(Pid, V) -> 
    gen_server:cast(?MODULE, {done, Pid, V}).

done_backprop(Pid) ->
    gen_server:cast(?MODULE, {done_backprop, Pid}).

save(Fn) ->
    gen_server:call(?MODULE, {save, Fn}).

to_term() ->
    gen_server:call(?MODULE, to_term).

to_c() ->
    gen_server:call(?MODULE, to_c).

load(Fn) ->
    (catch stop()),
    {ok, Pid} = gen_server:start_link({local, ?MODULE}, ?MODULE, {load, Fn}, []),
    Pid.

% ------------------------------
%  gen-server
% ------------------------------
init({load, Fn}) ->
    {ok, [T]} = file:consult(Fn),

    % create new ones
    AllPids = lists:map(fun (L) -> [nnperceptron:from_term(P) || P <- L] end, T),

    {D1, D2} = lists:foldl(fun (Lst, Acc) ->
                lists:foldl(
                    fun (Pid, {A1, A2}) -> 
                            I = nnperceptron:get_id(Pid),
                            {[{I, Pid} | A1], [{Pid, I} | A2]}
                    end, 
                    Acc, Lst) end,
        {[], []}, lists:flatten(AllPids)),
    gen_server:cast(?MODULE, link_up),
    {ok, #network{id2pid = dict:from_list(D1), 
                         pid2id = dict:from_list(D2), 
                         ni = length(hd(AllPids)),
                         outlayer = lists:last(AllPids), 
                         layers = AllPids}};
init({{NI, NHs, NO}, ActF}) ->
    Inputs = lists:map(fun (X) -> nnperceptron:new(identity, X) end, lists:seq(0, NI - 1)),
    Layers = lists:reverse(element(1, lists:foldl(fun (Num, {Acc, Start}) ->
                R = lists:map(fun (X) -> 
                            nnperceptron:new(ActF, X) 
                    end, lists:seq(Start, Num + Start - 1)),
                {[R | Acc], Start + Num}
        end, {[], NI}, lists:append(NHs, [NO])))),

    {Did2pid, Dpid2id} = lists:foldl(fun (Ls, Acc) ->
                update_pid_dict(Ls, Acc)
        end, {dict:new(), dict:new()}, [Inputs | Layers]),

    lists:foldl(fun (Next, ALayer) ->
                connect_layer(ALayer, Next),
                Next
        end, Inputs, Layers),
    {ok, #network{id2pid = Did2pid, pid2id = Dpid2id, ni = NI, 
                  outlayer = lists:last(Layers),
                  layers = [Inputs | Layers]}}.

handle_call({set_done, F}, _From, #network{ondone = F0} = State) ->
    {reply, F0, State#network{ondone = F}};
handle_call({set_done_backprop, F}, _From, #network{ondone_backprop = F0} = State) ->
    {reply, F0, State#network{ondone_backprop = F}};
handle_call({id2pid, Id}, _From, #network{id2pid = D} = State) ->
    R = case dict:find(Id, D) of
        {ok, V} -> V;
        _ -> undefined
    end,
    {reply, R, State};
handle_call({pid2id, Pid}, _From, #network{pid2id = D} = State) ->
    R = case dict:find(Pid, D) of
        {ok, V} -> V;
        _ -> undefined
    end,
    {reply, R, State};
handle_call(to_term, _From, State) ->
    {reply, serialize(State), State};
handle_call(to_c, _From, State) ->
    {reply, c_form(State), State};
handle_call({save, Fn}, _From, State) ->
    {ok, Fid} = file:open(Fn, [write]),
    io:fwrite(Fid, "~p.~n", [serialize(State)]),
    file:close(Fid),
    {reply, ok, State};
handle_call(stop, _From, #network{layers = Layers} =State) ->
    lists:foreach(fun (Layer) -> [nnperceptron:stop(Pid) || Pid <- Layer] end, Layers),
    {stop, normal, stopped, #network{}}.

handle_cast({pass, V}, #network{layers = [Inputs|_]} = State) ->
    lists:zipwith(fun (O, Ve) -> nnperceptron:pass(O, Ve) end, Inputs, V),
    {noreply, State#network{output_v = dict:new()}};
handle_cast({backprop, E}, #network{outlayer = Layer} = State) ->
    lists:zipwith(fun (Ele, Err) -> nnperceptron:backprop(Ele, Err) end, Layer, E),
    {noreply, State};
handle_cast({done, Pid, V}, #network{ondone = OnDone, output_v = D, outlayer =
        Layer} = State) ->
    %io:format("~p get ~p, ~p~n", [Pid, V, Layer]),
    D10 = dict:store(Pid, V, D),
    case dict:size(D10) == length(Layer) of
        true ->
            V10 = [dict:fetch(Pid, D10) || Pid <- Layer],
            if  is_function(OnDone) ->
                    OnDone(V10);
                is_pid(OnDone) ->
                    OnDone ! {done, V10};
                true ->
                    ok
            end,
            {noreply, State#network{output_v = undefined}};
        _ ->
            {noreply, State#network{output_v = D10}}
    end;
handle_cast({done_backprop, Pid}, #network{ondone_backprop = OnDone, ni = NI, backprop_set = S} = State) ->
    S10 = sets:add_element(Pid, S),
    false = sets:is_element(Pid, S),
    S20 = case sets:size(S10) == NI of
        true -> 
                if  is_function(OnDone) ->
                        OnDone();
                    is_pid(OnDone) ->
                        OnDone ! done_backprop;
                    true ->
                        ok
                end,
                sets:new();
        _ -> S10
    end,
    {noreply, State#network{backprop_set = S20}};
handle_cast(show, #network{layers = Layers} = State) ->
    lists:foreach(fun (Layer) -> io:format("Layer size: ~p~n", [length(Layer)]) end, Layers),
    {noreply, State};
handle_cast(link_up, #network{layers = Layers} = State) ->
     lists:foreach(fun (Layer) -> [nnperceptron:link_up(Pid) || Pid <- Layer] end, Layers),
     {noreply, State}.

handle_info(_Msg, State) ->
    {noreply, State}.

terminate(_Reason, #network{layers = Layers} = _State) ->
    lists:foreach(fun (Layer) -> [nnperceptron:stop(Pid) || Pid <- Layer] end, Layers),
    ok.    

code_change(_OldVsn, State, _Extra) -> {ok, State}.

update_pid_dict(L, {Did2pid0, Dpid2id0}) ->
    lists:foldl(fun (Pid, {Did2pid, Dpid2id}) -> 
                Id = nnperceptron:get_id(Pid),
                {dict:store(Id, Pid, Did2pid), dict:store(Pid, Id, Dpid2id)}
        end, {Did2pid0, Dpid2id0}, L).

select_one(M, [X | T], D) ->
    N = dict:fetch(X, D),
    case N < M of
        true -> {X, T, dict:store(X, N + 1, D)};
        _ -> select_one(M, T, D)
    end.

serialize(#network{layers = Layers} = _State) ->
    All = lists:map(fun (L) -> [nnperceptron:to_term(P) || P <- L] end, Layers),
    All.

c_form(#network{layers = Layers} = _State) ->
    O = lists:last(Layers),
    lists:zipwith(fun (I, P) -> lists:flatten(io_lib:format("f(~p) = ~s", [I,
                                                             nnperceptron:to_c(P)]))
                  end,
                  lists:seq(0, length(O) - 1), O).


lcm(A, B) -> (A * B) div gcd(A, B).

gcd(A, 0) -> A;
gcd(A, B) -> gcd(B, A rem B).

connect_layer(L1, L2) -> [nnperceptron:connect(X, Y) || X <- L1, Y <- L2].

%    % connecting: input -> hidden
%    Total = lcm(NI, NH),
%    N = Total div NI,
%    M = Total div NH,
%    lists:foldl(
%        fun (Pi, D) -> 
%                {_, D30} = lists:foldl(
%                    fun (_, {Candidates, D10}) ->
%                            {Pto, Candidates10, D20} = select_one(M, Candidates, D10),
%                            nnperceptron:connect(Pi, Pto),
%                            {Candidates10, D20}
%                    end, {Hiddens, D}, lists:seq(1, N)),
%                D30
%        end,
%        dict:from_list(lists:map(fun (Ph) -> {Ph, 0} end, Hiddens)),
%        Inputs),
%
%    % connecting: hidden -> output
%    lists:foreach(fun (Pfrom) -> nnperceptron:connect(Pfrom, O) end, Hiddens),
%
