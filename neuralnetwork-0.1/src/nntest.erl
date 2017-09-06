-module(nntest).

-define(all_activations, [sigmoid, tanh, arctan, rectified, softplus,
                          bent, sin, sinc, gaussian]).

-export([test_adder/1, test_naive/1]).

encode(N, Len) ->
    lists:map(fun (X) -> if X == N -> 1; true -> 0 end end, lists:seq(0,Len - 1)).

test_adder(N) ->
    [{F, test_adder(F, N)} || F <- ?all_activations].

test_v_to_n(CasesIn, F, N) ->
    (catch nnetwork:stop()),
    
    VMax = lists:max([I || {_, I} <- CasesIn]),

    (catch nnetwork:start({length(element(1, hd(CasesIn))), [3], VMax + 1}, F)),

    Cases = [{V, encode(T, VMax + 1)} || {V, T} <- CasesIn],
    lists:foreach(fun (_) ->
                lists:foreach(fun (I) ->
                            {V, R} = lists:nth(I, Cases),
                            E = lists:zipwith(fun (X, Y) -> X - Y end, 
                                R, nnetwork:sync_pass(V)),
                            nnetwork:sync_backprop(E)
                    end, lists:seq(1, length(Cases)))
        end, lists:seq(0, N)),
    [nnetwork:sync_pass(V) || {V, _} <- Cases].

test_adder(F, N) ->
    Cases = [{[0, 0], 0},
             {[0, 1], 1},
             {[1, 0], 1},
             {[1, 1], 2}],
    test_v_to_n(Cases, F, N).

train(N) ->
    (catch nnetwork:start({247, [30], 10})),
    {ok, [Cases0]} = file:consult("training.erl"),
    Cases = lists:map(fun ({C, T}) -> 
                              FL = lists:flatten(C),
                              A = 1 / lists:sum(FL), 
                              {[X * A || X <- FL], encode(T, 10)} 
                      end, Cases0),
    ntrainer:train(Cases, N).

train_naive(N) ->
    (catch nnetwork:start({2, [2], 2})),
    Cases0 = [
        {[0, 0], 1},
        {[1, 0], 0},
        {[1, 1], 0},
        {[0, 2], 0},
        {[2, 2], 0}
    ],
    Cases = lists:map(fun ({C, T}) -> {C, encode(T, 2)} end, Cases0),
    ntrainer:train(Cases, N).


