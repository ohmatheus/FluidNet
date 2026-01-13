#pragma once

#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>

#define PROFILE_SCOPE() ZoneScoped
#define PROFILE_SCOPE_NAMED(name) ZoneScopedN(name)
#define PROFILE_FRAME_MARK() FrameMark
#define PROFILE_SET_THREAD_NAME(name) tracy::SetThreadName(name)
#define PROFILE_PLOT(name, value) TracyPlot(name, value)
#define PROFILE_ZONE_TEXT(text, size) ZoneText(text, size)

#else

#define PROFILE_SCOPE()
#define PROFILE_SCOPE_NAMED(name)
#define PROFILE_FRAME_MARK()
#define PROFILE_SET_THREAD_NAME(name)
#define PROFILE_PLOT(name, value)
#define PROFILE_ZONE_TEXT(text, size)

#endif
