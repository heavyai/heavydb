### About ImgGui

Review the main readme for ImGui to understand the design goals and programming model: https://github.com/ocornut/imgui#readme

The current deps install uses the latest tagged release branch. There is a well maintained branch for ImGui, the docking branch, that adds separate window docking and multi-viewport capabilities, including dragging a window outside the main application window. This latter feature is apparently buggy on Linux, so the current deps install doesn't use it, saving it for future experimentation and possibly helping fix the bugs. You can read more about docking here https://github.com/ocornut/imgui/issues/2109

ImGui supports a large number of graphics and windowing APIs across Linux, Windows, and MacOS. It is intended for integration into tools and is very customizable. The flexibile design means it is not configured to build a standalone lib. This PR adds a new library `GfxImGui` using the components we need, as part of the `GfxDriver` `CMakeLists.txt`. It will only build if GLFW is found as we use the ImGui GLFW backend to automatically handle input events. If GLFW is found ImGui becomes a required dependency

### ImGuiBridge

ImGui integration is via the `ImGuiBridge` class, which handles loading fonts, render / upload the font texture atlas, and provide a few helpers such as an `Overlay` class.

`ImGuiBridge` supports 2 different "backend" classes to handle drawing the vertex/index buffers and uploading the font atlas:
 - Default backend wraps the standard ImGui Vulkan backend. While this works it manages resources outside our tracking
 - Custom backend uses the `GfxDriver` API to handle drawing, however this is still not rendering correctly

The dual backend setup is probably not permanent, but until the custom backend is working and we've had a chance to test out multiple viewports and docking which may present new challenges, it will be necessary. If we decide to keep it we can change it to use a virtual interface and factory

### Font scale in WSI apps using ImGui

ImGui does not automatically handle font-sizing, it's up to the application to scale fonts when adding them to the atlas, setting a glyph size in pixels. To handle hi-dpi we use a function in GLWF, `getWindowContentScale` to determine a scale multiplier for the glyph size. On hi-dpi displays this returns 2.0, doubling the font size. If you have fractional scaling enabled in Ubuntu display setting, it has no effect on the GLFW value or the displayed font.

Unfortunately I don't have an easy way to test on a normal dpi display (1080p). In the event the current scale isn't working for you, the program option `--font-scale` can be used to alter the font scale. A value of `1.0` uses the default size `2.0` will match the hi-dpi scale factor, etc. Fractional values work fine.

### GfxWSITest app

The `GfxWSITest` application serves as a template for building sandbox applications. It can be built and run using the `wsi_test` build target. To specify program options you will need to run `GfxWSITest` directly instead of using this target. Use `--help` to view options (`-h`, is used for image height).

The UI uses a tab panel for `Settings`, `Memory Summary`, and `Help`, and should serve as a starting template

### MultiSymbol app

This application is the procedural symbol rendering sandbox and will build and run automatically using the `multisymbol` build target. It uses the production shader templates directly enabling rapid iteration on changes to the symbol rendering pipeline. It also includes a rudamental symbol definition editor for interactive editing of individual symbol definitions. As is standard, the program option `--help` will list all program options. Once in the app, use the `u` hotkey to open the UI, which has a `Help` tab outlining interactive usage