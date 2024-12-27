using Pkg
using Revise
#Pkg.activate(".")
Pkg.activate(@__DIR__)
#run(`bash -c "module load gurobi"`)

ENV["GUROBI_HOME"] = "/nopt/nrel/apps/software/gurobi/gurobi1100/linux64"
ENV["PATH"] *= ":$ENV[\"GUROBI_HOME\"]/bin"
ENV["GRB_LICENSE_FILE"] = "/nopt/nrel/apps/software/gurobi/tlicense/gurobi.lic"  

using PowerSystems
using PowerSimulations
using PowerSystemCaseBuilder
using EasternInterconnection
using PowerNetworkMatrices
using HydroPowerSimulations
using StorageSystemsSimulations
using JuMP
using Dates
using Gurobi
using PowerSimulationsDecomposition
using Revise
using HiGHS
using SparseArrays
using LinearAlgebra
using KLU
using CSV, DataFrames

const HPS = HydroPowerSimulations
const EI = EasternInterconnection
const PSI = PowerSimulations
const SSS = StorageSystemsSimulations
const PSY = PowerSystems
const PNM = PowerNetworkMatrices
const PSB = PowerSystemCaseBuilder

include("YC_test_EI_test_function.jl")
include("ward_decompose.jl")
include("row_cache.jl")

case_name = "EI_2031_PCM"
#name = "EI_2031_Legacy_Reduced_PCM"
weather_year = 2009
load_year = 2035
EIPC_data = true
add_forecasts = false
add_reserves = true
skip_serialization = false

@time sys = EI.build_system(
                EI.EISystems,
                case_name;
                weather_year = weather_year,
                load_year = load_year,
                EIPC_data = EIPC_data,
                add_forecasts = add_forecasts,
                add_reserves = add_reserves,
                skip_serialization = skip_serialization,
                force_build = false,
                runchecks = false,
            )

EI.change_areas_to_regions!(sys)

branches = PNM.get_ac_branches(sys)
buses = PNM.get_buses(sys)
all_areas = PSY.get_components(PSY.Area, sys);
area = collect(all_areas)[2]    ###select the study area
internal_buses = sort([get_number(b) for b in buses if get_area(b) == area])   #####identify the internal buses

line_ax = [PSY.get_name(branch) for branch in branches]
bus_ax = [PSY.get_number(bus) for bus in buses]
axes = (line_ax, bus_ax)
M, bus_ax_ref = PNM.calculate_adjacency(branches, buses)
subnetworks = find_subnetworks(M, bus_ax)

max_length = maximum(length.(values(subnetworks)))
df_subnetworks = DataFrame()
for (key, values) in subnetworks
    # Convert set to vector and specify it can include Missing
    column_data = Vector{Union{Int64, Missing}}(collect(values))
    # Pad with missing if necessary
    if length(column_data) < max_length
        # Extend column_data with missing values to ensure all columns have the same length
        column_data = resize!(column_data, max_length)
        fill!(column_data[length(values)+1:end], missing)
    end
    df_subnetworks[!, string(key)] = column_data  # Use string(key) to name the column
end

### prepared calculation, including the identification of subnetwork and compute the ybus matrix of whole system
bus_ax = PSY.get_number.(buses)
axes = (bus_ax, bus_ax)
bus_lookup = PNM.make_ax_ref(bus_ax)
look_up = (bus_lookup, bus_lookup)
fixed_admittances = collect(PSY.get_components(PSY.FixedAdmittance, sys))
ybus = @time(PNM._buildybus(branches, buses, fixed_admittances))


#########################check the internal buses and subnetworks
internal_buses_set = Set(internal_buses)
subnetworks_sets = Dict{String, Set{Int64}}()
for col in names(df_subnetworks)
    cleaned_data = filter(x -> !ismissing(x), df_subnetworks[!, col])
    subnetworks_sets[col] = Set{Int64}(cleaned_data)
end

containing_subnetworks = Set{String}()
for (network_id, bus_set) in subnetworks_sets
    if !isempty(intersect(internal_buses_set, bus_set))
        push!(containing_subnetworks, network_id)
    end
end
num_subnetworks = length(containing_subnetworks)
###########################

################get the ybus of subnetwork############
subnetwork_ids = collect(keys(subnetworks))
studied_subnetwork_id = subnetwork_ids[7]
studied_subnetwork_buses = collect(subnetworks[studied_subnetwork_id])
# studied_subnetwork_buses_lookup = Dict(k => v for (k, v) in bus_lookup if k in studied_subnetwork_buses)
studied_subnetwork_buses_lookup = PNM.make_ax_ref(studied_subnetwork_buses)
studied_subnetwork_buses_ax = get.(Ref(bus_lookup), studied_subnetwork_buses, nothing)
studied_subnetwork_ybus = ybus[studied_subnetwork_buses_ax,studied_subnetwork_buses_ax]
studied_subnetwork_ybus
PSY.get_bus_numbers(sys)
###############################

##### only do the ward decomposition from whole ybus matrix to smaller ybus matrix####### about 11 seconds
ybus_study = @time(ward_decompose(sys,studied_subnetwork_ybus,studied_subnetwork_buses,internal_buses))

##### this function include the identification of subnetwork and compute the whole ybus matrix#####about 15 seconds
ybus_study = @time(ward_decompose(sys,internal_buses))

#######identify the line name based on the all branches and ybus_study, also identify the reference bus
to_b, fr_b = SparseArrays.findnz(ybus_study.data)
non_diag_indices = (fr_b .< to_b)
fr_b = fr_b[non_diag_indices]
to_b = to_b[non_diag_indices]

reverse_bus_lookup = Dict(value => key for (key, value) in ybus_study.lookup[1])
branch_map = Dict(
    (branch.arc.from.number, branch.arc.to.number) => branch.name for branch in branches
)
study_branch_names = Vector{String}(undef, length(fr_b))  ##get the name of branches in the study system. new branches will be named 'new'

for i in 1:length(fr_b)
    from_actual = reverse_bus_lookup[fr_b[i]]
    to_actual = reverse_bus_lookup[to_b[i]]
    branch_name = get(branch_map, (from_actual, to_actual), "new_$i")
    study_branch_names[i] = branch_name
end

study_line_ax_ref = PNM.make_ax_ref(study_branch_names)
A, ref_bus_positions = PNM.calculate_A_matrix(branches, buses)
bus_lookup
reverse_bus_lookup_whole = Dict(value => key for (key, value) in bus_lookup)

reference_bus = ybus_study.lookup[1][reverse_bus_lookup_whole[collect(ref_bus_positions)[1]]]
Set(reference_bus)
##### compute the PTDF######
study_ptdf = @time(PNM.VirtualPTDF(ybus_study,Set(reference_bus),study_branch_names,study_line_ax_ref));   ### compute the studied system
whole_ptdf = @time(PNM.VirtualPTDF(branches,buses));                                                  ### compute the whole system




using Test
passed_count = 0
failed_count = 0

lines_ptdf_not_same = []
study_ptdf
whole_ptdf
for i in study_ptdf.axes[1]
    if startswith(i, "new")
        continue
    end
    for j in study_ptdf.axes[2]
        # Perform the approximate equality test manually
        if isapprox(study_ptdf[i, j], whole_ptdf[i, j], atol=0.001)
            passed_count += 1
        else
            failed_count += 1
            push!(lines_ptdf_not_same,i)
            # println("Test failed at ($i, $j): $(study_ptdf[i, j]) vs $(whole_ptdf[i, j])")
        end
    end
end
lines_ptdf_not_same = unique(lines_ptdf_not_same)
println(passed_count)
println(failed_count)
