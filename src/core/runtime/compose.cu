#include "legion.h"

#include <memory>

#include "core/runtime/resource.h"
#include "core/task/exception.h"
#include "core/utilities/typedefs.h"
#include "default_mapper.h"
#include "legion/legion_mapping.h"
#include "legion/legion_c.h"

#include "compose.h"

namespace Legion{

    using namespace Legion;
    using namespace Legion::Mapping;


    __global__
    void myKernel(int r0l, int r0h, int r1l, int r1h, FieldAccessor<READ_WRITE,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t> > writeTo){
        int x = blockIdx.x * blockDim.x + threadIdx.x + r0l;
        int y = blockIdx.y * blockDim.y + threadIdx.y + r1l;

        if(x <= r0h && y <= r1h){
            Point<2> p = Point<2>(x,y);

            writeTo[p] = writeTo[p] * 2;
        }
    }
    
    __host__
    void fake_task(
        const Legion::Task *task, 
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx, Legion::Runtime *runtime
    )
    {
        Rect<2> rect = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
        
        FieldAccessor<READ_WRITE,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t> > writeTo(regions[0],1000);

        dim3 threads_per_block(16,16);
        dim3 num_blocks((rect.hi[0] - rect.lo[0] + threads_per_block.x) / threads_per_block.x, (rect.hi[1] - rect.lo[1] + threads_per_block.y) / threads_per_block.y);


        myKernel<<<num_blocks,threads_per_block>>>(rect.lo[0], rect.hi[0], rect.lo[1], rect.hi[1], writeTo);
    }
    
    __host__
    int ComposeClassC::registerFake(){
        Legion::Runtime* runtime = Legion::Runtime::get_runtime();
        
        int printOp = runtime->generate_dynamic_task_id();
        {
            TaskVariantRegistrar registrar(printOp, "designed_to_compose");
            registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
            runtime->register_task_variant<fake_task>(registrar);
        }

        return printOp;
    }
}