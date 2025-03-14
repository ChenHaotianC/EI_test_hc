"""
The Virtual Power Transfer Distribution Factor (VirtualPTDF) structure gathers
the rows of the PTDF matrix as they are evaluated on-the-go. These rows are
evalauted independently, cached in the structure and do not require the
computation of the whole matrix (therefore significantly reducing the
computational requirements).

The VirtualPTDF is initialized with no row stored.

The VirtualPTDF is indexed using branch names and bus numbers as for the PTDF
matrix.

# Arguments
- `K::KLU.KLUFactorization{Float64, Int}`:
        LU factorization matrices of the ABA matrix, evaluated by means of KLU
- `BA::SparseArrays.SparseMatrixCSC{Float64, Int}`:
        BA matric
- `ref_bus_positions::Set{Int}`:
        Vector containing the indexes of the columns of the BA matrix corresponding
        to the reference buses
- `dist_slack::Vector{Float64}`:
        Vector of weights to be used as distributed slack bus.
        The distributed slack vector has to be the same length as the number of buses.
- `axes<:NTuple{2, Dict}`:
        Tuple containing two vectors: the first one showing the branches names,
        the second showing the buses numbers. There is no link between the
        order of the vector of the branches names and the way the PTDF rows are
        stored in the cache.
- `lookup<:NTuple{2, Dict}`:
        Tuple containing two dictionaries, mapping the branches
        and buses with their enumerated indexes. The branch indexes refer to
        the key of the cache dictionary. The bus indexes refer to the position
        of the elements in the PTDF row stored.
- `temp_data::Vector{Float64}`:
        Temporary vector for internal use.
- `valid_ix::Vector{Int}`:
        Vector containing the row/columns indices of matrices related the buses
        which are not slack ones.
- `cache::RowCache`:
        Cache were PTDF rows are stored.
- `subnetworks::Dict{Int, Set{Int}}`:
        Dictionary containing the subsets of buses defining the different subnetwork of the system.
- `tol::Base.RefValue{Float64}`:
        Tolerance related to scarification and values to drop.
- `radial_network_reduction::RadialNetworkReduction`:
        Structure containing the radial branches and leaf buses that were removed
        while evaluating the matrix
"""
struct VirtualPTDF{Ax, L <: NTuple{2, Dict}} <: PowerNetworkMatrix{Float64}
    K::KLU.KLUFactorization{Float64, Int}
    BA::SparseArrays.SparseMatrixCSC{Float64, Int}
    ref_bus_positions::Set{Int}
    dist_slack::Vector{Float64}
    axes::Ax
    lookup::L
    temp_data::Vector{Float64}
    valid_ix::Vector{Int}
    cache::RowCache
    subnetworks::Dict{Int, Set{Int}}
    tol::Base.RefValue{Float64}
    radial_network_reduction::RadialNetworkReduction
end

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, array::VirtualPTDF)
    summary(io, array)
    isempty(array) && return
    println(io, ":")
    Base.print_array(io, array)
    return
end

"""
Builds the PTDF matrix from a group of branches and buses. The return is a
VirtualPTDF struct with an empty cache.

# Arguments
- `branches`:
        Vector of the system's AC branches.
- `buses::Vector{PSY.ACBus}`:
        Vector of the system's buses.

# Keyword Arguments
- `dist_slack::Vector{Float64} = Float64[]`:
        Vector of weights to be used as distributed slack bus.
        The distributed slack vector has to be the same length as the number of buses.
- `tol::Float64 = eps()`:
        Tolerance related to sparsification and values to drop.
- `max_cache_size::Int`:
        max cache size in MiB (inizialized as MAX_CACHE_SIZE_MiB).
- `persistent_lines::Vector{String}`:
        line to be evaluated as soon as the VirtualPTDF is created (initialized as empty vector of strings).
- `radial_network_reduction::RadialNetworkReduction`:
        Structure containing the radial branches and leaf buses that were removed
        while evaluating the matrix
"""
function VirtualPTDF(
    branches,
    buses::Vector{PSY.ACBus};
    dist_slack::Vector{Float64} = Float64[],
    tol::Float64 = eps(),
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    persistent_lines::Vector{String} = String[],
    radial_network_reduction::RadialNetworkReduction = RadialNetworkReduction(),
)
    if length(dist_slack) != 0
        @info "Distributed bus"
    end

    #Get axis names
    line_ax = [PSY.get_name(branch) for branch in branches]
    bus_ax = [PSY.get_number(bus) for bus in buses]
    axes = (line_ax, bus_ax)
    M, bus_ax_ref = calculate_adjacency(branches, buses)
    line_ax_ref = make_ax_ref(line_ax)
    look_up = (line_ax_ref, bus_ax_ref)
    A, ref_bus_positions = calculate_A_matrix(branches, buses)
    BA = calculate_BA_matrix(branches, bus_ax_ref)
    ABA = calculate_ABA_matrix(A, BA, ref_bus_positions)
    ref_bus_positions = find_slack_positions(buses)
    subnetworks = find_subnetworks(M, bus_ax)
    if length(subnetworks) > 1
        @info "Network is not connected, using subnetworks"
        subnetworks = assign_reference_buses!(subnetworks, ref_bus_positions, bus_ax_ref)
    end
    temp_data = zeros(length(bus_ax))

    if isempty(persistent_lines)
        empty_cache =
            RowCache(max_cache_size * MiB, Set{Int}(), length(bus_ax) * sizeof(Float64))
    else
        init_persistent_dict = Set{Int}(line_ax_ref[k] for k in persistent_lines)
        empty_cache =
            RowCache(
                max_cache_size * MiB,
                init_persistent_dict,
                length(bus_ax) * sizeof(Float64),
            )
    end

    return VirtualPTDF(
        klu(ABA),
        BA,
        ref_bus_positions,
        dist_slack,
        axes,
        look_up,
        temp_data,
        setdiff(1:length(temp_data), ref_bus_positions),
        empty_cache,
        subnetworks,
        Ref(tol),
        radial_network_reduction,
    )
end

function VirtualPTDF(
    Ybus,
    ref_bus_positions::Set{Int64},
    line_ax::Vector{String},
    line_ax_ref::Dict{String, Int64},
    dist_slack::Vector{Float64} = Float64[],
    tol::Float64 = eps(),
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    persistent_lines::Vector{String} = String[],
    radial_network_reduction::RadialNetworkReduction = RadialNetworkReduction(),
)
    if length(dist_slack) != 0
        @info "Distributed bus"
    end

    ybus = Ybus.data
    bus_ax = convert.(Int64, Ybus.axes[1])
    bus_ax_ref = Ybus.lookup[1]
    axes = (line_ax, bus_ax)
    look_up = (line_ax_ref, bus_ax_ref)

    #compute A
    number_branches = round(Int, (SparseArrays.nnz(ybus)-size(ybus,1))/2)
    A_I = repeat(1:number_branches, inner = 2)
    to_b, fr_b = SparseArrays.findnz(ybus)
    non_diag_indices = (fr_b .< to_b)
    fr_b = fr_b[non_diag_indices]
    to_b = to_b[non_diag_indices]
    A_J = vec(hcat(fr_b, to_b)')
    A_V = Int8.(repeat([1, -1], outer=number_branches))
    A = SparseArrays.sparse(A_I, A_J, A_V)

    #compute BA
    BA_V = imag.(getindex.(Ref(ybus), fr_b, to_b))
    BA_V = 1.0 ./ imag.(1.0 ./ getindex.(Ref(ybus), fr_b, to_b))
    BA_V = vcat((x -> [x, -x]).(BA_V)...)
    BA = SparseArrays.sparse(A_J,A_I, BA_V)

    #compute M
    # M = SparseArrays.spzeros(Int8, size(ybus))
    positive_direction_mask = imag.(ybus) .> 0
    negative_direction_mask = imag.(ybus) .< 0
    # @time(M[positive_direction_mask] .= 1)
    # @time(M[negative_direction_mask] .= -1)


    rows, cols = size(ybus)
    pos_indices = findall(positive_direction_mask)
    neg_indices = findall(negative_direction_mask)
    pos_row_indices = getindex.(pos_indices, 1)
    pos_col_indices = getindex.(pos_indices, 2)
    neg_row_indices = getindex.(neg_indices, 1)
    neg_col_indices = getindex.(neg_indices, 2)
    pos_values = fill(1, length(pos_indices))
    neg_values = fill(-1, length(neg_indices))
    all_row_indices = vcat(pos_row_indices, neg_row_indices)
    all_col_indices = vcat(pos_col_indices, neg_col_indices)
    all_values = vcat(pos_values, neg_values)
    M = SparseArrays.sparse(all_row_indices, all_col_indices, all_values, rows, cols, Int8)


    row_sums = sum(abs.(ybus), dims=2)
    isolated = row_sums .== 0
    M[LinearAlgebra.diagind(M)] .= .!isolated

    subnetworks = find_subnetworks(M, bus_ax)

    #compute ABA
    ABA = calculate_ABA_matrix(A, BA, ref_bus_positions)

    if length(subnetworks) > 1
        @info "Network is not connected, using subnetworks"
        subnetworks = assign_reference_buses!(subnetworks, ref_bus_positions, bus_ax_ref)
    end
    temp_data = zeros(length(bus_ax))

    if isempty(persistent_lines)
        empty_cache =
            RowCache(max_cache_size * MiB, Set{Int}(), length(bus_ax) * sizeof(Float64))
    else
        init_persistent_dict = Set{Int}(line_ax_ref[k] for k in persistent_lines)
        empty_cache =
            RowCache(
                max_cache_size * MiB,
                init_persistent_dict,
                length(bus_ax) * sizeof(Float64),
            )
    end

    return VirtualPTDF(
        klu(ABA),
        BA,
        ref_bus_positions,
        dist_slack,
        axes,
        look_up,
        temp_data,
        setdiff(1:length(temp_data), ref_bus_positions),
        empty_cache,
        subnetworks,
        Ref(tol),
        radial_network_reduction,
    )
end

"""
Builds the Virtual PTDF matrix from a system. The return is a VirtualPTDF
struct with an empty cache.

# Arguments
- `sys::PSY.System`:
        PSY system for which the matrix is constructed

# Keyword Arguments
- `dist_slack::Vector{Float64}=Float64[]`:
        vector of weights to be used as distributed slack bus.
        The distributed slack vector has to be the same length as the number of buse
- `reduce_radial_branches::Bool=false`:
        if True the matrix will be evaluated discarding
        all the radial branches and leaf buses (optional, default value is false)
- `kwargs...`:
        other keyword arguments used by VirtualPTDF
"""
function VirtualPTDF(
    sys::PSY.System;
    dist_slack::Vector{Float64} = Float64[],
    reduce_radial_branches::Bool = false,
    kwargs...,
)
    if reduce_radial_branches
        A = IncidenceMatrix(sys)
        dist_slack, rb = redistribute_dist_slack(dist_slack, A)
    else
        rb = RadialNetworkReduction()
    end
    branches = get_ac_branches(sys, rb.radial_branches)
    buses = get_buses(sys, rb.bus_reduction_map)
    return VirtualPTDF(
        branches,
        buses;
        dist_slack = dist_slack,
        radial_network_reduction = rb,
        kwargs...,
    )
end

# Overload Base functions

"""
Checks if the any of the fields of VirtualPTDF is empty.
"""
function Base.isempty(vptdf::VirtualPTDF)
    for name in fieldnames(typeof(vptdf))
        if name == :dist_slack && !isempty(getfield(vptdf, name))
            @debug "Field dist_slack has default value: " *
                   string(getfield(vptdf, name)) * "."
            return false
        elseif (name in [:cache, :radial_branches]) && !isempty(getfield(vptdf, name))
            @debug "Field " * string(name) * " not defined."
            return false
        end
    end
    return true
end

"""
Gives the size of the whole PTDF matrix, not the number of rows stored.
"""
Base.size(vptdf::VirtualPTDF) = size(vptdf.BA)

"""
Gives the cartesian indexes of the PTDF matrix (same as the BA one).
"""
Base.eachindex(vptdf::VirtualPTDF) = CartesianIndices(size(vptdf.BA))

if isdefined(Base, :print_array) # 0.7 and later
    Base.print_array(io::IO, X::VirtualPTDF) = "VirtualPTDF"
end

function _getindex(
    vptdf::VirtualPTDF,
    row::Int,
    column::Union{Int, Colon},
)
    # check if value is in the cache
    if haskey(vptdf.cache, row)
        return vptdf.cache.temp_cache[row][column]
    else
        # evaluate the value for the PTDF column
        # Needs improvement
        valid_ix = vptdf.valid_ix
        lin_solve = KLU.solve!(vptdf.K, Vector(vptdf.BA[valid_ix, row]))
        buscount = size(vptdf, 1)

        if !isempty(vptdf.dist_slack) && length(vptdf.ref_bus_positions) != 1
            error(
                "Distibuted slack is not supported for systems with multiple reference buses.",
            )
        elseif isempty(vptdf.dist_slack) && length(vptdf.ref_bus_positions) < buscount
            for i in eachindex(valid_ix)
                vptdf.temp_data[valid_ix[i]] = lin_solve[i]
            end
            vptdf.cache[row] = deepcopy(vptdf.temp_data)
        elseif length(vptdf.dist_slack) == buscount
            for i in eachindex(valid_ix)
                vptdf.temp_data[valid_ix[i]] = lin_solve[i]
            end
            slack_array = vptdf.dist_slack / sum(vptdf.dist_slack)
            slack_array = reshape(slack_array, buscount)
            vptdf.cache[row] =
                deepcopy(vptdf.temp_data .- dot(vptdf.temp_data, slack_array))
        else
            error("Distributed bus specification doesn't match the number of buses.")
        end

        if get_tol(vptdf) > eps()
            vptdf.cache[row] = deepcopy(sparsify(vptdf.cache[row], get_tol(vptdf)))
        end

        return vptdf.cache[row][column]
    end
end

"""
Gets the value of the element of the PTDF matrix given the row and column indices
corresponding to the branch and buses one respectively. If `column` is a Colon then
the entire row is returned.

# Arguments
- `vptdf::VirtualPTDF`:
        VirtualPTDF struct where to evaluate and store the row values.
- `row`:
        Branch index.
- `column`:
        Bus index. If Colon then get the values of the whole row.
"""
function Base.getindex(vptdf::VirtualPTDF, row, column)
    row_, column_ = to_index(vptdf, row, column)
    return _getindex(vptdf, row_, column_)
end

# Define for ambiguity resolution
function Base.getindex(vptdf::VirtualPTDF, row::Integer, column::Integer)
    return _getindex(vptdf, row, column)
end

"""
!!! STILL TO IMPLEMENT !!!
"""
Base.setindex!(::VirtualPTDF, _, idx...) = error("Operation not supported by VirtualPTDF")

"""
!!! STILL TO IMPLEMENT !!!
"""
Base.setindex!(::VirtualPTDF, _, ::CartesianIndex) =
    error("Operation not supported by VirtualPTDF")

get_ptdf_data(mat::VirtualPTDF) = mat.cache.temp_cache

function get_branch_ax(ptdf::VirtualPTDF)
    return ptdf.axes[1]
end

function get_bus_ax(ptdf::VirtualPTDF)
    return ptdf.axes[2]
end

""" Gets the tolerance used for sparsifying the rows of the VirtualPTDF matrix"""
function get_tol(vptdf::VirtualPTDF)
    return vptdf.tol[]
end
