#include "compose.h"
#include "core/utilities/linearize.h"
// #include "compose.cu"

namespace legate {
namespace mapping {
    //For now slice_by_legate presumes identity projector for key partition
    void slice_as_legate(
        const Legion::Task& task,
        const legate::mapping::MachineDesc& machine_desc,
        const Machine& machine, 
        const Legion::Mapping::Mapper::SliceTaskInput& input,
        Legion::Mapping::Mapper::SliceTaskOutput& output)
    {
        output.slices.reserve(input.domain.get_volume());

        //Don't hard program TOC_PROC for obvious reasons
        auto local_range = machine.slice(TaskTarget::GPU, machine_desc);

        Domain sharding_domain = task.index_domain;
        auto lo = sharding_domain.lo();
        auto hi = sharding_domain.hi();

        uint32_t total_tasks_count = linearize(lo, hi, hi) + 1;

        uint32_t start_proc_id = machine_desc.processor_range().low;

        for (Domain::DomainPointIterator itr(input.domain); itr; itr++) {
            // auto p = key_functor->project_point(itr.p, sharding_domain);
            auto p = itr.p;
            uint32_t idx =
            linearize(lo, hi, p) * local_range.total_proc_count() / total_tasks_count + start_proc_id;
            output.slices.push_back(
            Legion::Mapping::Mapper::TaskSlice(Domain(itr.p, itr.p), local_range[idx], false /*recurse*/, false /*stealable*/));
        }
    }
}
}

namespace Legion{
    using namespace Legion;
    using namespace Legion::Mapping;

    std::vector<std::vector<int>> directions = {{-1,0},{1,0},{0,-1},{0,1}};

    void fill_region(
        const Task *task, 
        const std::vector<PhysicalRegion> &regions,
        Context ctx, Runtime *runtime)
    {
        const FieldAccessor<WRITE_DISCARD,int32_t,2> acc_x(regions[0], 1000);
        Rect<2> elem_rect = runtime->get_index_space_domain(
            ctx, 
            task->regions[0].region.get_index_space()
        );

        int i = 0;
        for (PointInRectIterator<2> pir(elem_rect); pir(); pir++)
        {
            acc_x[*pir] = i++;
        }
    }

    void print_region(
        const Task *task, 
        const std::vector<PhysicalRegion> &regions,
        Context ctx, Runtime *runtime)
    {
        const FieldAccessor<READ_ONLY,double,2> acc_input(regions[0], 0);
        const FieldAccessor<WRITE_DISCARD,double,2> acc_output(regions[1], 0);
        Rect<2> elem_rect = runtime->get_index_space_domain(
            ctx,
            task->regions[0].region.get_index_space()
        );

        for (PointInRectIterator<2> pir(elem_rect); pir(); pir++)
        {
            Point<2> current_point = *pir;
            double accumulate = 0;
            for(uint i = 0; i < directions.size(); i++){
            Point<2> neighbor_point = Point<2>(current_point[0] + directions[i][0], current_point[1] + directions[i][1]);
            if (elem_rect.contains(neighbor_point))
                accumulate += acc_input[neighbor_point];
            acc_output[current_point] = accumulate / 4;
            }
            printf("value is %f with points (%llu, %llu) \n", accumulate / 4, current_point.x, current_point.y);
        }
    }
  
    int ComposeClassC::returnFour(){
        return 4;
    }

    int ComposeClassC::registerFill(){
        Legion::Runtime* runtime = Legion::Runtime::get_runtime();
        
        int fillOp = runtime->generate_dynamic_task_id();
        fillOp += 1;
        {
        TaskVariantRegistrar registrar(fillOp, "hello_world");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        runtime->register_task_variant<fill_region>(registrar);
        }

        return fillOp;
    }

    int ComposeClassC::registerPrint(){
        Legion::Runtime* runtime = Legion::Runtime::get_runtime();
        
        int printOp = runtime->generate_dynamic_task_id();
        {
        TaskVariantRegistrar registrar(printOp, "hello_world");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        runtime->register_task_variant<print_region>(registrar);
        }

        return printOp;
    }

    int ComposeClassC::registeredID = 5;
    int ComposeClassC::nodeCount = 0;

    void ComposeClassC::oldRegisterMapper(Legion::Machine machine,
                                            Legion::Runtime* runtime,
                                            const std::set<Processor>& local_procs){
        
        std::string temp = "supercrazylib";
        auto new_id = runtime->generate_library_mapper_ids(temp.c_str(), 1);

        auto base_mapper = new ComposingMapper(machine, runtime, *local_procs.begin());
        
        base_mapper->fullMachineDesc.preferred_target = legate::mapping::TaskTarget::GPU;
        //Get GPU count
        Machine::ProcessorQuery gpuQuery(machine);
        gpuQuery.only_kind(Processor::TOC_PROC);
        legate::mapping::ProcessorRange gpuRange(0, gpuQuery.count(), gpuQuery.count() / nodeCount);
        base_mapper->fullMachineDesc.processor_ranges[legate::mapping::TaskTarget::GPU] = gpuRange;

        //Get OMP count
        Machine::ProcessorQuery ompQuery(machine);
        ompQuery.only_kind(Processor::OMP_PROC);
        legate::mapping::ProcessorRange ompRange(0, ompQuery.count(), ompQuery.count() / nodeCount);
        base_mapper->fullMachineDesc.processor_ranges[legate::mapping::TaskTarget::OMP] = ompRange;

        //Get CPU count
        Machine::ProcessorQuery cpuQuery(machine);
        cpuQuery.only_kind(Processor::LOC_PROC);
        legate::mapping::ProcessorRange cpuRange(0, gpuQuery.count(), gpuQuery.count() / nodeCount);
        base_mapper->fullMachineDesc.processor_ranges[legate::mapping::TaskTarget::CPU] = cpuRange;

        runtime->add_mapper(new_id, base_mapper);
        
        registeredID = new_id;
    }

    void ComposeClassC::newRegisterMapper(int nC){
        nodeCount = nC;
        Legion::Runtime::perform_registration_callback(ComposeClassC::oldRegisterMapper, true);
    }

    int ComposeClassC::getID(){
        return registeredID;
    }

    void ComposingMapper::select_task_options(const MapperContext ctx,
                                        const Task &task,
                                        TaskOptions &output)
    {
        DefaultMapper::select_task_options(ctx, task, output);
    }

    void ComposingMapper::slice_task(const MapperContext      ctx,
                                    const Task&              task,
                                    const SliceTaskInput&    input,
                                            SliceTaskOutput&   output)
    {
        legate::mapping::slice_as_legate(task, fullMachineDesc, machine, input, output);
        return;
        // Everything I had written below eventually pushed out by slice_as_legate
    }

}