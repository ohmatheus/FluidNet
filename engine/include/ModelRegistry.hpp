#pragma once

#include <string>
#include <vector>

namespace FluidNet
{

struct ModelInfo
{
    std::string name;
    std::string path;
    bool isLoaded{false};
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

private:
    std::vector<ModelInfo> m_models;
    int m_currentIndex{0};
};

}
