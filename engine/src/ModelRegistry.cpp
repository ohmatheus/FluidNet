#include "ModelRegistry.hpp"
#include <algorithm>
#include <filesystem>
#include <stdexcept>

namespace FluidNet
{

void ModelRegistry::initialize(const std::string& modelsFolder)
{
    m_models.clear();

    if (!std::filesystem::exists(modelsFolder))
    {
        throw std::runtime_error("Models folder does not exist: " + modelsFolder);
    }

    // Scan folder for .onnx files
    for (const auto& entry : std::filesystem::directory_iterator(modelsFolder))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".onnx")
        {
            std::string name = entry.path().stem().string();
            std::string path = entry.path().string();
            m_models.push_back({name, path, false});
        }
    }

    // Sort alphabetically
    std::sort(m_models.begin(), m_models.end(),
              [](const ModelInfo& a, const ModelInfo& b) { return a.name < b.name; });

    if (m_models.empty())
    {
        throw std::runtime_error("No .onnx models found in folder: " + modelsFolder);
    }

    m_currentIndex = 0;
}

void ModelRegistry::selectModel(int index)
{
    if (index >= 0 && index < static_cast<int>(m_models.size()))
    {
        m_currentIndex = index;
    }
}

void ModelRegistry::nextModel()
{
    m_currentIndex = (m_currentIndex + 1) % static_cast<int>(m_models.size());
}

void ModelRegistry::prevModel()
{
    m_currentIndex = (m_currentIndex - 1 + static_cast<int>(m_models.size())) %
                     static_cast<int>(m_models.size());
}

const ModelInfo& ModelRegistry::getCurrentModel() const
{
    return m_models[m_currentIndex];
}

}
