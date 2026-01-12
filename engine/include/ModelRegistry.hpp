#pragma once

#include <string>
#include <vector>

namespace FluidNet
{

enum class ModelPrecision
{
    FP32,
    FP16,
    INT8
};

struct ModelInfo
{
    std::string name;
    std::string pathFP32;
    std::string pathFP16;
    std::string pathINT8;
    std::string relativeDir;
    std::string displayName;
    bool isLoaded{false};
    bool hasFP16Variant{false};
    bool hasINT8Variant{false};
};

class ModelRegistry
{
public:
    ModelRegistry() = default;

    void initialize(const std::string& modelsFolder);

    void selectModel(int index);
    void nextModel();
    void prevModel();

    const ModelInfo& getCurrentModel() const;
    const std::vector<ModelInfo>& getModels() const
    {
        return m_models;
    }
    int getCurrentIndex() const
    {
        return m_currentIndex;
    }

    std::string getModelPath(int index, ModelPrecision precision) const;
    std::string getCurrentModelPath(ModelPrecision precision) const;
    std::vector<ModelPrecision> getAvailablePrecisions(int index) const;

private:
    std::vector<ModelInfo> m_models;
    int m_currentIndex{0};
};

}
