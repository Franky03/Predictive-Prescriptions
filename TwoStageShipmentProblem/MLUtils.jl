module MLUtils

export MinMaxScaler, min_max_fit!, min_max_transform!, min_max_fit_transform!, min_max_inverse_transform!

mutable struct MinMaxScaler
    min_vals::Union{Vector{Float64}, Nothing}
    max_vals::Union{Vector{Float64}, Nothing}
end

function MinMaxScaler()
    return MinMaxScaler(nothing, nothing)
end

function min_max_fit!(scaler::MinMaxScaler, X::Matrix{Float64})
    """
    Calculate the minimum and maximum values for each feature in the dataset
    """
    scaler.min_vals = minimum(X, dims=1)[:]
    scaler.max_vals = maximum(X, dims=1)[:]
    return scaler
end

function min_max_transform!(scaler::MinMaxScaler, X::Matrix{Float64})
    """
    Normalize the dataset X using the stored minimum and maximum values.
    """
    if isnothing(scaler.min_vals) || isnothing(scaler.max_vals)
        throw(ArgumentError("The scaler needs to be fitted with fit! before applying transform!"))
    end
    # Normalize each element
    return (X .- scaler.min_vals') ./ (scaler.max_vals' .- scaler.min_vals')
end

function min_max_fit_transform!(scaler::MinMaxScaler, X::Matrix{Float64})
    """
    Fit the scaler and transform the input data.
    """
    fit!(scaler, X)
    return transform!(scaler, X)
end

function min_max_inverse_transform!(scaler::MinMaxScaler, X_scaled::Matrix{Float64})
    """
    Reverse the transformation back to the original values.
    """
    if isnothing(scaler.min_vals) || isnothing(scaler.max_vals)
        throw(ArgumentError("The scaler needs to be fitted with fit! before applying inverse_transform!"))
    end
    return X_scaled .* (scaler.max_vals' .- scaler.min_vals') .+ scaler.min_vals'
end

end