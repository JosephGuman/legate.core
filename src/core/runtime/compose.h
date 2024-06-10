#pragma once

#include "legion.h"

#include <memory>

#include "core/runtime/resource.h"
#include "core/task/exception.h"
#include "core/utilities/typedefs.h"
#include "default_mapper.h"
#include "legion/legion_mapping.h"
#include "legion/legion_c.h"
#include "core/mapping/machine.h"

namespace Legion{
  class ComposeClassC{
    public:
      static int returnFour();
      static int registerFill();
      static int registerPrint();
      static int registerFake();
      
      static void oldRegisterMapper(Legion::Machine machine,
                                           Legion::Runtime* runtime,
                                           const std::set<Processor>& local_procs);
      
      static void newRegisterMapper(int nodeCount);
      static int getID();
      static int registeredID;
      static int nodeCount;
  };


  using namespace Legion;
  using namespace Legion::Mapping;

  class ComposingMapper : public Legion::Mapping::DefaultMapper {
    public:
      ComposingMapper(Machine machine,
          Runtime *rt, Processor local) : DefaultMapper(rt->get_mapper_runtime(), machine, local), machine() {};
      virtual void slice_task(const MapperContext ctx,
                              const Task& task,
                              const SliceTaskInput& input,
                                    SliceTaskOutput& output);

      void select_task_options(const MapperContext ctx,
                                const Task &task,
                                TaskOptions &output);

    public:
      legate::mapping::Machine machine;
      legate::mapping::MachineDesc fullMachineDesc;
  };

}
