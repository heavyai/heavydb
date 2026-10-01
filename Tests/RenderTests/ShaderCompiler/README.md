### ShaderCompiler test data maintenance

Automation for maintaining input data and results is not in place yet. Input data consists of `ShaderManager::Builder` serialization files (JSON), which are deserialized by the test. These builders are compiled to Spir-V, which is then checked two ways: it must pass `spirv-val`, and the `ShaderReflection` extracted from it must match the matching `.reflect` file in the results data set.

The results used to be `.spv` blobs compared byte for byte. That could only ever answer "did the bytes change", never "did the meaning change", so every toolchain bump forced a blind regeneration and the test went quiet exactly when it was most needed. The `.reflect` files are text, so when one changes the diff shows which binding, offset or location moved and a human can judge whether that is acceptable.

Note that reflection covers the shader interface, not the contents of function bodies. A change in what a shader computes, with its interface intact, will not show up here; `spirv-val` and the image-comparison render tests are the backstop for that.

* The Manual dataset files are created by loading the matching vega into vega-editor and producing the required artifacts. These could be replaced with more comprehensive coverage files based on be-render-tests.
* The Renderer dataset files are the non-dynamic shaders created during server startup.

Fast symbol Manual files are generated via 2 different builder files:
* `FastSymbolTest.json` vega generates `fastSymbolTemplate.vert.builder` and `fastSymbolTemplate.frag.builder`
* `FastSymbolTest_angle.json` vega generates `fastSymbolTemplate_passthru.vert.builder` and `fastSymbolTemplate.geom.builder`

To generate artifacts:

* Specify a path to save the artifacts to using `OMNISCI_shader_artifact_path`. This is generally useful so may as well add it to your .bashrc
* Paste the vega json from a file in `Tests/RenderTests/ShaderCompiler/Manual/Vega` into `vega-editor`. The artifacts will automatically be written into a subfolder in `OMNISCI_shader_artifact_path`
* Copy the `.builder` files into `Inputs`

To recreate the `.reflect` files from existing `.builder` files:

* Run `ShaderCompilerTest --regenerate-reflect`
* **Always read the resulting diff.** Regenerating is only correct once you have decided the new bindings, offsets and locations are what you want. A diff you cannot explain is a bug, not a baseline.
* Setting `OMNISCI_always_save_shader_artifacts=reflect` writes the same format as a `.reflection` artifact for any shader the server compiles, which is the quickest way to see the reflection for a shader that has no test case.