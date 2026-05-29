# --------------------------------------------------------------------------
# Inverse rendering utilities: gradient-based optimization of scene params.
# Uses ForwardDiff for gradient computation.
# --------------------------------------------------------------------------

using ForwardDiff

"""
Gradient descent optimizer for inverse rendering.
Optimizes `params` to minimize the loss between rendered image and target.

Arguments:
- initial_params: starting parameter vector
- target_image: goal image (H×W×3 array)
- render_fn: params -> image (must be ForwardDiff-compatible)
- loss_fn: (rendered, target) -> scalar loss
- lr: learning rate
- n_iters: number of optimization steps
- verbose: print progress

Returns: (optimized_params, loss_history)
"""
function inverse_render_optimize(initial_params::Vector{Float64},
                                 target_image::Array{Float64, 3},
                                 render_fn::Function,
                                 loss_fn::Function;
                                 lr=0.01,
                                 n_iters=100,
                                 verbose=true)
    params = copy(initial_params)
    n = length(params)
    loss_history = Float64[]

    for iter in 1:n_iters
        # Compute loss and gradient
        function objective(p)
            img = render_fn(p)
            loss_fn(img, target_image)
        end

        current_loss = objective(params)
        push!(loss_history, current_loss)

        grad = ForwardDiff.gradient(objective, params)

        # Gradient descent step
        params .-= lr .* grad

        if verbose && (iter % 10 == 0 || iter == 1)
            @info "Iter $iter/$n_iters: loss = $(round(current_loss, sigdigits=6))"
        end
    end

    return params, loss_history
end

"""
Adam optimizer for inverse rendering.
Better convergence than vanilla gradient descent.
"""
function inverse_render_adam(initial_params::Vector{Float64},
                            target_image::Array{Float64, 3},
                            render_fn::Function,
                            loss_fn::Function;
                            lr=0.01,
                            β1=0.9, β2=0.999,
                            ε=1e-8,
                            n_iters=100,
                            verbose=true)
    params = copy(initial_params)
    n = length(params)
    m = zeros(n)  # first moment
    v = zeros(n)  # second moment
    loss_history = Float64[]

    for iter in 1:n_iters
        function objective(p)
            img = render_fn(p)
            loss_fn(img, target_image)
        end

        current_loss = objective(params)
        push!(loss_history, current_loss)

        grad = ForwardDiff.gradient(objective, params)

        # Adam update
        m .= β1 .* m .+ (1 - β1) .* grad
        v .= β2 .* v .+ (1 - β2) .* grad .^ 2
        m_hat = m ./ (1 - β1^iter)
        v_hat = v ./ (1 - β2^iter)
        params .-= lr .* m_hat ./ (sqrt.(v_hat) .+ ε)

        if verbose && (iter % 10 == 0 || iter == 1)
            @info "Adam iter $iter/$n_iters: loss = $(round(current_loss, sigdigits=6))"
        end
    end

    return params, loss_history
end

"""
Compute numerical (finite difference) gradients for validation.
"""
function numerical_gradient(f, params::Vector{Float64}; δ=1e-5)
    n = length(params)
    grad = zeros(n)
    f0 = f(params)
    for i in 1:n
        p_plus = copy(params)
        p_plus[i] += δ
        grad[i] = (f(p_plus) - f0) / δ
    end
    return grad
end
