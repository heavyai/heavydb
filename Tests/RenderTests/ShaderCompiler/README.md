### ShaderCompiler test data maintenance

Automation for maintaining input data and results is not in place yet. Input data consists of `ShaderManager::Builder` serialization files (JSON), which are deserialized by the test. These builders are processed to produce Spir-V which is then compared to the matching results data set:

* The Manual dataset files are created by loading the matching vega into vega-editor and producing the required artifacts. These could be replaced with more comprehensive coverage files based on be-render-tests.
* The Renderer dataset files are the non-dynamic shaders created during server startup.

Fast symbol Manual files are generated via 2 different builder files:
* `FastSymbolTest.json` vega generates `fastSymbolTemplate.vert.builder` and `fastSymbolTemplate.frag.builder`
* `FastSymbolTest_angle.json` vega generates `fastSymbolTemplate_passthru.vert.builder` and `fastSymbolTemplate.geom.builder`

To generate artifacts:

* Specify a path to save the artifacts to using `OMNISCI_shader_artifact_path`. This is generally useful so may as well add it to your .bashrc
* Paste the vega json from a file in `Tests/RenderTests/ShaderCompiler/Manual/Vega` into `vega-editor`. The `.builder` and `.spv` files will automatically be written into a subfolder in `OMNISCI_shader_artifact_path`
* Copy the files into the `Inputs` (`.builder`) and `Results` (`.spv`) directories

To recreate just the `.spv` files from existing `.builder` files:

* Run `ShaderCompilerTest --regenerate-spv`
* This should generally only be used when the toolchain is updated (e.g. glslang changes). For anything that potentially impacts builder structure prefer using the `.spv` files generated with the `.builder` files