#pragma once

#include <atomic>
#include <vector>

namespace FluidNet
{

struct SceneMaskSnapshot
{
    std::vector<float> emitterMask;
    std::vector<float> colliderMask;
    std::vector<float> velocityImpulseX;
    std::vector<float> velocityImpulseY;
    int gridResolution{0};
};

class SceneState
{
public:
    explicit SceneState(int gridResolution);

    void paintEmitter(int gridX, int gridY, int brushSize);
    void paintCollider(int gridX, int gridY, int brushSize);
    void paintVelocityImpulse(int gridX, int gridY, float velocityX, float velocityY,
                              int brushSize);
    void erase(int gridX, int gridY, int brushSize);
    void clear();
    void decayVelocityImpulses(float decayFactor);
    void commitSnapshot();

    const std::atomic<SceneMaskSnapshot*>* getSnapshotAtomic() const
    {
        return &m_snapshotForSim;
    }
    const std::vector<float>& getEmitterLayer() const
    {
        return m_emitterLayer;
    }
    const std::vector<float>& getColliderLayer() const
    {
        return m_colliderLayer;
    }
    int getGridResolution() const
    {
        return m_gridResolution;
    }

private:
    std::vector<float> m_emitterLayer;
    std::vector<float> m_colliderLayer;
    std::vector<float> m_velocityImpulseX;
    std::vector<float> m_velocityImpulseY;
    int m_gridResolution;

    SceneMaskSnapshot m_snapshots[3];
    std::atomic<int> m_nextWriteIndex{0};
    std::atomic<SceneMaskSnapshot*> m_snapshotForSim{nullptr};

    void paintCircle_(std::vector<float>& layer, int centerX, int centerY, int radius, float value);
};

}
