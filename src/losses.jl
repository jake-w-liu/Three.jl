# --------------------------------------------------------------------------
# Image-space loss functions for inverse rendering.
# All pure Julia, ForwardDiff compatible.
# --------------------------------------------------------------------------

"""
L2 (MSE) loss between two images.
Images are Array{T, 3} of size (H, W, C).
"""
function loss_mse(image::Array{T, 3}, target::Array{S, 3}) where {T, S}
    @assert size(image) == size(target)
    H, W, C = size(image)
    total = zero(promote_type(T, S))
    n = H * W * C
    for c in 1:C
        for j in 1:W
            for i in 1:H
                d = image[i, j, c] - target[i, j, c]
                total += d * d
            end
        end
    end
    return total / n
end

"""
L1 loss between two images.
"""
function loss_l1(image::Array{T, 3}, target::Array{S, 3}) where {T, S}
    @assert size(image) == size(target)
    H, W, C = size(image)
    total = zero(promote_type(T, S))
    n = H * W * C
    for c in 1:C
        for j in 1:W
            for i in 1:H
                total += abs(image[i, j, c] - target[i, j, c])
            end
        end
    end
    return total / n
end

"""
Structural Similarity Index (SSIM) loss.
Returns 1 - SSIM (so that minimizing this maximizes SSIM).
Simplified single-channel average SSIM.
"""
function loss_ssim(image::Array{T, 3}, target::Array{T, 3};
                   window_size=7, C1=0.01^2, C2=0.03^2) where T
    H, W, C = size(image)
    hw = window_size ÷ 2
    ssim_sum = zero(T)
    count = 0

    for c in 1:C
        for j in (hw+1):(W-hw)
            for i in (hw+1):(H-hw)
                # Local means
                μx = zero(T)
                μy = zero(T)
                n = 0
                for dj in -hw:hw
                    for di in -hw:hw
                        μx += image[i+di, j+dj, c]
                        μy += target[i+di, j+dj, c]
                        n += 1
                    end
                end
                μx /= n
                μy /= n

                # Local variances and covariance
                σx2 = zero(T)
                σy2 = zero(T)
                σxy = zero(T)
                for dj in -hw:hw
                    for di in -hw:hw
                        dx = image[i+di, j+dj, c] - μx
                        dy = target[i+di, j+dj, c] - μy
                        σx2 += dx * dx
                        σy2 += dy * dy
                        σxy += dx * dy
                    end
                end
                σx2 /= (n - 1)
                σy2 /= (n - 1)
                σxy /= (n - 1)

                ssim_val = (2*μx*μy + C1) * (2*σxy + C2) /
                           ((μx^2 + μy^2 + C1) * (σx2 + σy2 + C2))
                ssim_sum += ssim_val
                count += 1
            end
        end
    end

    mean_ssim = count > 0 ? ssim_sum / count : one(T)
    return one(T) - mean_ssim
end

"""
Silhouette IoU loss (Intersection over Union).
Compares binary silhouettes extracted from images.
`threshold` separates foreground from background.
"""
function loss_silhouette_iou(image::Array{T, 3}, target::Array{T, 3};
                              threshold=0.01) where T
    H, W, _ = size(image)

    # Convert to grayscale silhouettes via soft thresholding
    intersection = zero(T)
    union_val = zero(T)

    for j in 1:W
        for i in 1:H
            # Brightness as max channel
            img_val = max(image[i, j, 1], image[i, j, 2], image[i, j, 3])
            tgt_val = max(target[i, j, 1], target[i, j, 2], target[i, j, 3])

            # Soft occupancy
            img_occ = sigmoid_approx((img_val - threshold) * 50)
            tgt_occ = sigmoid_approx((tgt_val - threshold) * 50)

            intersection += img_occ * tgt_occ
            union_val += img_occ + tgt_occ - img_occ * tgt_occ
        end
    end

    iou = intersection / max(union_val, T(1e-8))
    return one(T) - iou
end
