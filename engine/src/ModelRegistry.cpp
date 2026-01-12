#include "ModelRegistry.hpp"
#include <algorithm>
#include <filesystem>
#include <map>
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

    // First pass: collect FP32 models only
    std::map<std::string, ModelInfo> modelsMap;

    for (const auto& entry : std::filesystem::recursive_directory_iterator(modelsFolder))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".onnx" &&
            entry.path().string().find(".onnx.meta") == std::string::npos)
        {
            std::string stem = entry.path().stem().string();

            if ((stem.size() > 5 && stem.substr(stem.size() - 5) == "_int8") ||
                (stem.size() > 5 && stem.substr(stem.size() - 5) == "_fp16"))
            {
                continue;
            }

            std::string name = stem;
            std::string path = entry.path().string();
            std::filesystem::path relativePath =
                std::filesystem::relative(entry.path().parent_path(), modelsFolder);
            std::string relativeDir = relativePath.string();
            std::string displayName = relativeDir.empty() ? name : relativeDir + "/" + name;

            ModelInfo info{name, path, "", "", relativeDir, displayName, false, false, false};
            modelsMap[path] = info;
        }
    }

    // Second pass: find FP16 and INT8 variants
    for (auto& [fp32Path, modelInfo] : modelsMap)
    {
        std::filesystem::path fp32PathObj(fp32Path);
        std::filesystem::path parentPath = fp32PathObj.parent_path();
        std::string baseStem = fp32PathObj.stem().string();

        std::filesystem::path fp16Path = parentPath / (baseStem + "_fp16.onnx");
        if (std::filesystem::exists(fp16Path))
        {
            modelInfo.pathFP16 = fp16Path.string();
            modelInfo.hasFP16Variant = true;
        }

        std::filesystem::path int8Path = parentPath / (baseStem + "_int8.onnx");
        if (std::filesystem::exists(int8Path))
        {
            modelInfo.pathINT8 = int8Path.string();
            modelInfo.hasINT8Variant = true;
        }
    }

    // Convert to vector and sort
    for (const auto& [_, info] : modelsMap)
    {
        m_models.push_back(info);
    }

    // Sort by relative directory first, then by name
    std::sort(m_models.begin(), m_models.end(),
              [](const ModelInfo& a, const ModelInfo& b)
              {
                  if (a.relativeDir != b.relativeDir)
                      return a.relativeDir < b.relativeDir;
                  return a.name < b.name;
              });

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

std::string ModelRegistry::getModelPath(int index, ModelPrecision precision) const
{
    if (index < 0 || index >= static_cast<int>(m_models.size()))
    {
        return "";
    }

    const ModelInfo& model = m_models[index];

    switch (precision)
    {
    case ModelPrecision::FP16:
        if (model.hasFP16Variant && !model.pathFP16.empty())
        {
            return model.pathFP16;
        }
        break; // Fallback to FP32

    case ModelPrecision::INT8:
        if (model.hasINT8Variant && !model.pathINT8.empty())
        {
            return model.pathINT8;
        }
        break; // Fallback to FP32

    case ModelPrecision::FP32:
    default:
        break; // Use FP32
    }

    return model.pathFP32;
}

std::string ModelRegistry::getCurrentModelPath(ModelPrecision precision) const
{
    return getModelPath(m_currentIndex, precision);
}

std::vector<ModelPrecision> ModelRegistry::getAvailablePrecisions(int index) const
{
    std::vector<ModelPrecision> precisions;

    if (index < 0 || index >= static_cast<int>(m_models.size()))
    {
        return precisions;
    }

    const ModelInfo& model = m_models[index];

    // FP32 always available
    precisions.push_back(ModelPrecision::FP32);

    if (model.hasFP16Variant)
    {
        precisions.push_back(ModelPrecision::FP16);
    }

    if (model.hasINT8Variant)
    {
        precisions.push_back(ModelPrecision::INT8);
    }

    return precisions;
}

}
