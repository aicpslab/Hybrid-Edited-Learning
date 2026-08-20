classdef MLPModel < handle
    %% MLPModel  Simple Multi-Layer Perceptron for baseline comparison
    %   Usage:
    %     mlp = MLPModel(dim_in, dim_out, [128, 64])
    %     mlp.train(X_train, Y_train, X_val, Y_val, lr, epochs, batch)
    %     Y_pred = mlp.forward(X_test)

    properties
        sizes; weights; biases; n_total;
    end

    methods
        function obj = MLPModel(dim_in, dim_out, hidden_sizes)
            if nargin < 3, hidden_sizes = [128, 64]; end
            obj.sizes = [dim_in, hidden_sizes, dim_out];
            obj.weights = {}; obj.biases = {};
            obj.n_total = 0;
            for i = 1:length(obj.sizes)-1
                s = sqrt(2.0 / obj.sizes(i));
                obj.weights{i} = single(randn(obj.sizes(i), obj.sizes(i+1)) * s);
                obj.biases{i} = single(zeros(1, obj.sizes(i+1)));
                obj.n_total = obj.n_total + numel(obj.weights{i});
            end
        end

        function Y = forward(obj, X)
            a = X;
            for i = 1:length(obj.weights)-1
                a = max(0, a * obj.weights{i} + obj.biases{i});  % ReLU
            end
            Y = a * obj.weights{end} + obj.biases{end};
        end

        function [tlh, vlh, best_vl] = train(obj, X, Y, Xv, Yv, lr, epochs, batch, verbose)
            if nargin < 9, verbose = false; end
            n = size(X,1); nb = max(1, floor(n/batch));
            mW = cellfun(@(w) zeros(size(w),'single'), obj.weights, 'UniformOutput', false);
            vW = cellfun(@(w) zeros(size(w),'single'), obj.weights, 'UniformOutput', false);
            mb_ = cellfun(@(b) zeros(size(b),'single'), obj.biases, 'UniformOutput', false);
            vb_ = cellfun(@(b) zeros(size(b),'single'), obj.biases, 'UniformOutput', false);
            b1 = 0.9; b2 = 0.999; eps_v = 1e-8; t_cnt = 0;
            best_vl = inf; bW = obj.weights; bB = obj.biases;
            tlh = zeros(1, epochs); vlh = zeros(1, epochs);

            for ep = 1:epochs
                idx = randperm(n); Xs = X(idx,:); Ys = Y(idx,:);
                el = 0.0;
                for bi = 1:nb
                    st = (bi-1)*batch+1; en = min(st+batch-1, n);
                    Xb = Xs(st:en,:); Yb = Ys(st:en,:);
                    % Forward
                    acts = {Xb}; a = Xb;
                    for i = 1:length(obj.weights)-1
                        a = max(0, a * obj.weights{i} + obj.biases{i}); acts{end+1} = a;
                    end
                    Yp = a * obj.weights{end} + obj.biases{end}; acts{end+1} = Yp;
                    err = Yp - Yb; el = el + mean(err(:).^2);
                    % Backward
                    d = err / size(Xb,1);
                    for i = length(obj.weights):-1:1
                        dW = acts{i}' * d; db = sum(d, 1);
                        t_cnt = t_cnt + 1;
                        mW{i} = b1*mW{i} + (1-b1)*dW; vW{i} = b2*vW{i} + (1-b2)*dW.^2;
                        mb_{i} = b1*mb_{i} + (1-b1)*db; vb_{i} = b2*vb_{i} + (1-b2)*db.^2;
                        mW_h = mW{i}/(1-b1^t_cnt); vW_h = vW{i}/(1-b2^t_cnt);
                        mb_h = mb_{i}/(1-b1^t_cnt); vb_h = vb_{i}/(1-b2^t_cnt);
                        obj.weights{i} = obj.weights{i} - lr * mW_h ./ (sqrt(vW_h) + eps_v);
                        obj.biases{i} = obj.biases{i} - lr * mb_h ./ (sqrt(vb_h) + eps_v);
                        if i > 1
                            d = (d * obj.weights{i}') .* (acts{i} > 0);
                        end
                    end
                end
                el = el / nb; tlh(ep) = el;
                vl = mean((obj.forward(Xv) - Yv).^2, 'all'); vlh(ep) = vl;
                if vl < best_vl
                    best_vl = vl; bW = obj.weights; bB = obj.biases;
                end
                if verbose && mod(ep, 40) == 0
                    fprintf('    ep %d: train=%.4e val=%.4e\n', ep, el, vl);
                end
            end
            obj.weights = bW; obj.biases = bB;
        end
    end
end
