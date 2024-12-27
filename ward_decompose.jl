"""
structure and functions for power systems analysis based on Ward's decomposition
"""

"""
Data related to the considered Ward decomposition is here stored.
Given the set of buses under study, the original Ybus can be divided into 
sub matrices belongin to internal (I), boundary buses (B) and external (E) buses.
       |Y_ii  Y_ib   0  |
       |                |
 Ybus =|Y_bi  Y_bb  Y_be|
       |                |
       | 0    Y_eb  Y_ee|

# Arguments
- `y_equ_boundary::Matrix{ComplexF64}`:
        the product of Y_bb - Y_be * inv(Y_ee) * Y_eb
- `ybus_study::Matrix{ComplexF64}`:
        the whole Ybus matrix of the equavilant system
"""

struct Ybus_study{Ax, L <: NTuple{2, Dict}} <: PNM.PowerNetworkMatrix{ComplexF64}
    data::SparseArrays.SparseMatrixCSC{ComplexF64, Int}
    axes::Ax
    lookup::L
end
function ward_decompose(
    sys::PSY.System,
    ybus_matrix_input::SparseMatrixCSC{ComplexF64, Int64},
    all_buses::Vector{Int64},
    # lookup_input::Dict{Int64, Int64},
    internal_buses::Vector{Int64};
    # line_lookup::Dict{String, Int64};
)
    branches = PNM.get_ac_branches(sys)
    # buses = PSY.get_bus_numbers(sys)
    buses = all_buses
    external_buses = Vector()
    boundary_buses = Vector()

    for b in branches
        if !in(b.arc.from.number, internal_buses) && in(b.arc.to.number, internal_buses)
            push!(boundary_buses, b.arc.from.number)
        end
        if in(b.arc.from.number, internal_buses) && !in(b.arc.to.number, internal_buses)
            push!(boundary_buses, b.arc.to.number)
        end

    end
    boundary_buses = sort(unique(boundary_buses))                                         #boundary buses   
    external_buses = sort(setdiff(buses, union(internal_buses, boundary_buses)))          #external buses                 

    # drop the resistance and make the ybus matrix singular    
    # ybus = PNM.Ybus(sys);
    # ybus_matrix = ybus.data;
    ###############
    # buses = PNM.get_buses(sys)
    # bus_ax = PSY.get_number.(buses)
    # axes = (bus_ax, bus_ax)
    # bus_lookup = PNM.make_ax_ref(bus_ax)
    # look_up = lookup_input
    # fixed_admittances = collect(PSY.get_components(PSY.FixedAdmittance, sys))
    # ybus_matrix = PNM._buildybus(branches, buses, fixed_admittances)
    ybus_matrix = ybus_matrix_input
    # buses = PSY.get_bus_numbers(sys)
    #################
    
    ybus_matrix = map(x -> x != 0.0 ? -1/imag(1/x)*im : x, ybus_matrix);
    row_sums = sum(ybus_matrix, dims=2) - diag(ybus_matrix);
    ybus_matrix[diagind(ybus_matrix)] = -row_sums;

    # get the mapping index, for comparing the ptdf before and after ward decomposition
    bus_lookup = PNM.make_ax_ref(buses) #create a mapping for the bus index

    mapped_internal_buses = [bus_lookup[bus] for bus in internal_buses]
    mapped_boundary_buses = [bus_lookup[bus] for bus in boundary_buses]
    mapped_external_buses = [bus_lookup[bus] for bus in external_buses]

    # divide the ybus into y_ii, y_bb, y_ee,....
    y_ee = ybus_matrix[mapped_external_buses, mapped_external_buses];
    y_eb = ybus_matrix[mapped_external_buses, mapped_boundary_buses];
    y_be = ybus_matrix[mapped_boundary_buses, mapped_external_buses];
    y_bb = ybus_matrix[mapped_boundary_buses, mapped_boundary_buses];
    y_ii = ybus_matrix[mapped_internal_buses, mapped_internal_buses];
    y_ib = ybus_matrix[mapped_internal_buses, mapped_boundary_buses];
    y_bi = ybus_matrix[mapped_boundary_buses, mapped_internal_buses];

    # compute the equavilant boundary ybus
    y_equ_boundary = y_bb - y_be * KLU.solve!(klu(y_ee), Matrix(y_eb));

    # construct the ybus of study system
    # study_buses = sort(union(mapped_internal_buses, mapped_boundary_buses));
    study_buses = union(mapped_internal_buses, mapped_boundary_buses);
    num_study_buses = length(study_buses);
    ybus_study = spzeros(ComplexF64, num_study_buses, num_study_buses); # study Ybus matrix
    # Create a mapping from bus number to index in the new matrix
    bus_lookup_study = PNM.make_ax_ref(study_buses)

    # Create indices for boundary and internal buses in ybus_study
    boundary_indices = map(bus -> bus_lookup_study[bus], mapped_boundary_buses);
    internal_indices = map(bus -> bus_lookup_study[bus], mapped_internal_buses);

    # construct ybus_study with y_equ_boundary, ybus_internal, y_ib, and y_bi
    ybus_study[boundary_indices, boundary_indices] .= y_equ_boundary;
    ybus_study[internal_indices, internal_indices] .= y_ii;
    ybus_study[internal_indices, boundary_indices] .= y_ib;
    ybus_study[boundary_indices, internal_indices] .= y_bi;

    study_bus_ax = union(internal_buses, boundary_buses)
    axe = (study_bus_ax, study_bus_ax)
    study_bus_lookup = PNM.make_ax_ref(study_bus_ax)
    look_up = (study_bus_lookup, study_bus_lookup)



    return (Ybus_study(ybus_study, axe, look_up))
end



function ward_decompose(
    sys::PSY.System,
    internal_buses::Vector{Int64};
)
    branches = PNM.get_ac_branches(sys)
    buses = PNM.get_buses(sys)
    fixed_admittances = collect(PSY.get_components(PSY.FixedAdmittance, sys))
    whole_ybus = @time(PNM._buildybus(branches, buses, fixed_admittances))

    bus_ax = PSY.get_number.(buses)
    bus_lookup = PNM.make_ax_ref(bus_ax)

    # line_ax = [PSY.get_name(branch) for branch in branches]
    bus_ax = [PSY.get_number(bus) for bus in buses]
    # axes = (line_ax, bus_ax)
    subnetworks = find_subnetworks(whole_ybus, bus_ax)
    intersection_sizes = Dict(key => length(intersect(subnetworks[key], internal_buses)) for key in keys(subnetworks))
    max_key = findmax(intersection_sizes)[2]
    studied_subnetwork_buses = collect(subnetworks[max_key])

    # studied_subnetwork_buses_lookup = Dict(k => v for (k, v) in bus_lookup if k in studied_subnetwork_buses)
    studied_subnetwork_buses_ax = get.(Ref(bus_lookup), studied_subnetwork_buses, nothing)
    ybus_matrix = whole_ybus[studied_subnetwork_buses_ax,studied_subnetwork_buses_ax]

    # buses = PSY.get_bus_numbers(sys)
    buses = studied_subnetwork_buses
    external_buses = Vector()
    boundary_buses = Vector()

    @time(for b in branches
        if !in(b.arc.from.number, internal_buses) && in(b.arc.to.number, internal_buses)
            push!(boundary_buses, b.arc.from.number)
        end
        if in(b.arc.from.number, internal_buses) && !in(b.arc.to.number, internal_buses)
            push!(boundary_buses, b.arc.to.number)
        end
    end)
    boundary_buses = sort(unique(boundary_buses))                                         #boundary buses   
    external_buses = sort(setdiff(buses, union(internal_buses, boundary_buses)))          #external buses                 
    
    ybus_matrix = map(x -> x != 0.0 ? -1/imag(1/x)*im : x, ybus_matrix);
    row_sums = sum(ybus_matrix, dims=2) - diag(ybus_matrix);
    ybus_matrix[diagind(ybus_matrix)] = -row_sums;

    # get the mapping index, for comparing the ptdf before and after ward decomposition
    bus_lookup = PNM.make_ax_ref(buses) #create a mapping for the bus index

    mapped_internal_buses = [bus_lookup[bus] for bus in internal_buses]
    mapped_boundary_buses = [bus_lookup[bus] for bus in boundary_buses]
    mapped_external_buses = [bus_lookup[bus] for bus in external_buses]

    # divide the ybus into y_ii, y_bb, y_ee,....
    y_ee = ybus_matrix[mapped_external_buses, mapped_external_buses];
    y_eb = ybus_matrix[mapped_external_buses, mapped_boundary_buses];
    y_be = ybus_matrix[mapped_boundary_buses, mapped_external_buses];
    y_bb = ybus_matrix[mapped_boundary_buses, mapped_boundary_buses];
    y_ii = ybus_matrix[mapped_internal_buses, mapped_internal_buses];
    y_ib = ybus_matrix[mapped_internal_buses, mapped_boundary_buses];
    y_bi = ybus_matrix[mapped_boundary_buses, mapped_internal_buses];

    # compute the equavilant boundary ybus
    y_equ_boundary = @time(y_bb - y_be * KLU.solve!(klu(y_ee), Matrix(y_eb)));

    # construct the ybus of study system
    # study_buses = sort(union(mapped_internal_buses, mapped_boundary_buses));
    study_buses = union(mapped_internal_buses, mapped_boundary_buses);
    num_study_buses = length(study_buses);
    ybus_study = spzeros(ComplexF64, num_study_buses, num_study_buses); # study Ybus matrix
    # Create a mapping from bus number to index in the new matrix
    bus_lookup_study = PNM.make_ax_ref(study_buses)

    # Create indices for boundary and internal buses in ybus_study
    boundary_indices = map(bus -> bus_lookup_study[bus], mapped_boundary_buses);
    internal_indices = map(bus -> bus_lookup_study[bus], mapped_internal_buses);

    # construct ybus_study with y_equ_boundary, ybus_internal, y_ib, and y_bi
    ybus_study[boundary_indices, boundary_indices] .= y_equ_boundary;
    @time(ybus_study[internal_indices, internal_indices] = y_ii);
    # @time(internal_idx = internal_indices .+ 1)
    # @time(ybus_study[internal_idx, internal_idx] = y_ii)
    ybus_study[internal_indices, boundary_indices] .= y_ib;
    ybus_study[boundary_indices, internal_indices] .= y_bi;

    study_bus_ax = union(internal_buses, boundary_buses)
    axe = (study_bus_ax, study_bus_ax)
    study_bus_lookup = PNM.make_ax_ref(study_bus_ax)
    look_up = (study_bus_lookup, study_bus_lookup)



    return (Ybus_study(ybus_study, axe, look_up))
end
