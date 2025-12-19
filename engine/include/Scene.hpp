#pragma once

namespace FluidNet
{

class Scene
{
public:
    virtual ~Scene() = default;

    virtual void onInit() = 0;
    virtual void onShutdown() = 0;
    virtual void onUpdate(float deltaTime) = 0;
    virtual void render() = 0;
    virtual void onRenderUI() = 0;
    virtual void onKeyPress(int key, int scancode, int action, int mods) = 0;
    virtual void restart() = 0;

protected:
    Scene() = default;
};

}
