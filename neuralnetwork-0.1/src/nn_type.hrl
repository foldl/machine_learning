
-record(perceptron, 
    {
        id,

        inputs = [],    % [Id, ...]: ids of perceptrons that connect to the input of this one
        outputs = [],   % [Id, ...]: ids of perceptrons that connect to the output of this one
        weights = [0.5],

        % cache of this round of stimulation
        net = undefined,            % weights * input
        last = undefined,

        % a callback for adding input to this one: accept(Perceptron) -> {ok, Weight} | false
        accept = fun (_) -> {ok, rand:uniform() - 0.5} end,     

        fft,                    % feedforword type

        inputstate = [], % avaliable inputs
        laststate,
        sensitivity = [],
        total = 0       % length of inputs
    }).
