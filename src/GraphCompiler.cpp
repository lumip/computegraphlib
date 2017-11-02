#include "GraphCompiler.hpp"
#include "nodes/Node.hpp"

GraphCompiler::GraphCompiler()
    : _context(std::map<std::string, const float*>())
{

}

GraphCompiler::~GraphCompiler() { }

void GraphCompiler::Compile(const std::vector<Node::const_ptr>& nodes, const InputDataMap& inputData)
{
    GraphCompilationContext context(inputData);
    for (auto node : nodes)
    {
        node->Compile(&context);
    }
}
