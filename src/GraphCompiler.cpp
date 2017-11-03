#include "GraphCompiler.hpp"
#include "nodes/Node.hpp"

GraphCompiler::GraphCompiler()
    : _context(InputDataMap())
{

}

GraphCompiler::~GraphCompiler() { }

void GraphCompiler::Compile(const std::vector<ConstNodePtr>& nodes, const InputDataMap& inputData)
{
    GraphCompilationContext context(inputData);
    for (auto node : nodes)
    {
        node->Compile(&context);
    }
}
