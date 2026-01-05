#include "SceneState.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace FluidNet
{

SceneState::SceneState(int gridResolution) : m_gridResolution(gridResolution)
{
    size_t size = gridResolution * gridResolution;
    m_emitterLayer.resize(size, 0.0f);
    m_colliderLayer.resize(size, 0.0f);
    m_velocityImpulseX.resize(size, 0.0f);
    m_velocityImpulseY.resize(size, 0.0f);

    for (int i = 0; i < 3; ++i)
    {
        m_snapshots[i].emitterMask.resize(size, 0.0f);
        m_snapshots[i].colliderMask.resize(size, 0.0f);
        m_snapshots[i].velocityImpulseX.resize(size, 0.0f);
        m_snapshots[i].velocityImpulseY.resize(size, 0.0f);
        m_snapshots[i].gridResolution = gridResolution;
    }

    m_snapshotForSim.store(&m_snapshots[0]);
}

void SceneState::paintEmitter(int gridX, int gridY, int brushSize)
{
    paintCircle_(m_emitterLayer, gridX, gridY, brushSize, 1.0f);
}

void SceneState::paintCollider(int gridX, int gridY, int brushSize)
{
    paintCircle_(m_colliderLayer, gridX, gridY, brushSize, 1.0f);
}

void SceneState::paintVelocityImpulse(int gridX, int gridY, float velocityX, float velocityY, int brushSize)
{
    for (int dy = -brushSize; dy <= brushSize; ++dy)
    {
        for (int dx = -brushSize; dx <= brushSize; ++dx)
        {
            int x = gridX + dx;
            int y = gridY + dy;

            if (x < 0 || x >= m_gridResolution || y < 0 || y >= m_gridResolution)
                continue;

            float dist = std::sqrt(static_cast<float>(dx * dx + dy * dy));
            if (dist > brushSize)
                continue;

            int idx = y * m_gridResolution + x;
            m_velocityImpulseX[idx] += velocityX;
            m_velocityImpulseY[idx] += velocityY;
        }
    }
}

void SceneState::erase(int gridX, int gridY, int brushSize)
{
    paintCircle_(m_emitterLayer, gridX, gridY, brushSize, 0.0f);
    paintCircle_(m_colliderLayer, gridX, gridY, brushSize, 0.0f);
}

void SceneState::clear()
{
    std::fill(m_emitterLayer.begin(), m_emitterLayer.end(), 0.0f);
    std::fill(m_colliderLayer.begin(), m_colliderLayer.end(), 0.0f);
    std::fill(m_velocityImpulseX.begin(), m_velocityImpulseX.end(), 0.0f);
    std::fill(m_velocityImpulseY.begin(), m_velocityImpulseY.end(), 0.0f);
}

void SceneState::decayVelocityImpulses(float decayFactor)
{
    for (size_t i = 0; i < m_velocityImpulseX.size(); ++i)
    {
        m_velocityImpulseX[i] *= decayFactor;
        m_velocityImpulseY[i] *= decayFactor;
    }
}

void SceneState::commitSnapshot()
{
    int writeIdx = m_nextWriteIndex.load();
    m_nextWriteIndex.store((writeIdx + 1) % 3);

    SceneMaskSnapshot* snapshot = &m_snapshots[writeIdx];

    size_t byteSize = m_emitterLayer.size() * sizeof(float);
    std::memcpy(snapshot->emitterMask.data(), m_emitterLayer.data(), byteSize);
    std::memcpy(snapshot->colliderMask.data(), m_colliderLayer.data(), byteSize);
    std::memcpy(snapshot->velocityImpulseX.data(), m_velocityImpulseX.data(), byteSize);
    std::memcpy(snapshot->velocityImpulseY.data(), m_velocityImpulseY.data(), byteSize);

    m_snapshotForSim.store(snapshot);
}

void SceneState::paintCircle_(std::vector<float>& layer, int centerX, int centerY, int radius, float value)
{
    for (int dy = -radius; dy <= radius; ++dy)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            int x = centerX + dx;
            int y = centerY + dy;

            if (x < 0 || x >= m_gridResolution || y < 0 || y >= m_gridResolution)
                continue;

            float dist = std::sqrt(static_cast<float>(dx * dx + dy * dy));
            if (dist > radius)
                continue;

            int idx = y * m_gridResolution + x;

            if (value > 0.0f)
            {
                layer[idx] = value;
            }
            else
            {
                layer[idx] = value;
            }
        }
    }
}

}
