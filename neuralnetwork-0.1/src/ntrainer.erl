-module(ntrainer).

-export([train_to/2, train/1, train/2]).

-define(ERROR_CONVERGENCE, 0.01).

train_to(V, T) ->
    S = self(),
    spawn(fun () ->
                nnetwork:set_on_done(self()),
                nnetwork:set_on_done_backprop(self()),
                nnetwork:pass(V),
                process_loop(20, {V, T, S})
        end),
    receive 
        {train_done, ok} -> ok;
        X -> X
    after 3000 -> error
    end.

train(Cases) -> 
    train(Cases, 2000).

train(Cases, N) -> 
    spawn(fun () ->
                nnetwork:set_on_done(self()),
                nnetwork:set_on_done_backprop(self()),
                process_cases_loop(N, {Cases, {infinite, undefined}})
        end).

save({BestErr, BestNW}) ->
    Fn = lists:flatten(io_lib:format("~.2f.erl", [BestErr])),
    {ok, Fid} = file:open(Fn, [write]),
    io:fwrite(Fid, "~p.~n", [BestNW]),
    file:close(Fid).

process_cases_loop(N, {Cases, {BestErr, BestNW}}) -> 
    % select max error
    Err = lists:max([v_error(nnetwork:sync_pass(V), T) || {V, T} <- Cases]),

    if 
        abs(Err - BestErr) =< ?ERROR_CONVERGENCE * Err -> 
            io:format("done, N = ~p, MaxError = ~p~n", [N, Err]);
        N == 0 ->
            io:format("all steps used. MaxError = ~p, BestErr = ~p~n", [Err, BestErr]),
            save({BestErr, BestNW});
        true ->
            if N rem 100 == 0 -> io:format("progress, MaxError = ~p~n", [Err]); true -> ok end,
            %io:format("progress, MaxError = ~p, for ~n", [Err]),
            
            lists:foreach(fun ({V, T}) -> train_to(V, T) end, Cases),
            process_cases_loop(N - 1, {Cases, {Err, nnetwork:to_term()}})
    end.

process_loop(N, {V, T, S}) ->
    receive
        {done, R} ->
            nnetwork:backprop(subtract(T, R)),
            process_loop(N, {V, T, S});
        done_backprop ->
            if
                N > 0 -> 
                    nnetwork:pass(V),
                    process_loop(N - 1, {V, T, S});
                true -> 
                    S ! {train_done, ok}
            end;
        X -> 
            io:format("unknown ~p~n", [X]),
            S ! {train_done, {error, X}}
    end.        

norm(L) -> math:sqrt(lists:sum([X * X || X <- L])).
subtract(L1, L2) -> lists:zipwith(fun (X, Y) -> X - Y end, L1, L2).

v_error(L1, L2) -> math:sqrt(lists:sum(lists:zipwith(fun (X, Y) -> math:pow(X - Y, 2) end,
                                           L1, L2))).
