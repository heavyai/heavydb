---
name: glslang to slang migration
overview: Replace glslang with Slang and upgrade to Vulkan 1.4, sequenced as independently shippable PRs that each preserve full functionality, starting with a test safety net and ending with optional shader source migration. Originally eight; PR 5 has since split into 5a, 5b and 5c, and the SDK bump moved to the front after it turned out not to be gated.
todos:
  - id: spike-sdk
    content: "PR 0a: Spike the SDK bump to collect actual errors. Done: the tree builds and runs correctly on Vulkan SDK 1.4.363.0 with glslang 16.6.0, with the private headers still in place"
    status: completed
  - id: spike-slang-glsl
    content: "PR 0b: Compile all shader sources through slangc's GLSL compatibility mode (-allow-glsl) and count failures"
    status: pending
  - id: test-safety-net
    content: "PR 1: Replace byte-exact SPIR-V golden comparison in ShaderCompilerTest.cpp with ShaderReflection text goldens plus spirv-val; add reflection-dump artifact via existing kSpvReflect hook. Landed as 95fb438ec0, all 9 goldens generated and the suite passes"
    status: completed
  - id: hygiene
    content: "PR 2: Fix kShaderStageCount, harden glslang error-log scraping, remove vestigial multi-set scaffolding in ShaderRedecorator and ShaderReflection::validateSet, and fix the three ShaderReflection defects found during PR 1 (clear and operator= both skip fragment_shader_output_locations_, and the suppressed move constructor)"
    status: pending
  - id: remove-tshaderirutils
    content: "PR 3: Delete TShaderIRUtils and reimplement subroutine binding as a post-assembly textual pass over final_glsl, after includes are prepended"
    status: pending
  - id: remove-iomapresolver
    content: "PR 4: Link all stages of a material into a single glslang::TProgram using public mapIO(nullptr, nullptr); delete IoMapResolver, sentinels, and ShaderRedecorator's allocation logic, preserving clash diagnostics"
    status: pending
  - id: sdk-bump
    content: "PR 5a: Move the dependency tree to Vulkan SDK 1.4.363.0 with Slang built alongside glslang. Done ahead of PRs 3 and 4, which turned out not to gate it"
    status: completed
  - id: version-retarget
    content: "PR 5b: Retarget the compiler to Vulkan 1.3 / SPIR-V 1.6, which is free since the device floor is already 1.3, and normalize the VulkanPhysicalDevice version comparison. The instance bump to 1.4 is a separate deployment decision that raises the driver floor and gains the compiler nothing; nothing 1.4 promotes is used here"
    status: pending
  - id: drop-private-headers
    content: "PR 5c: After PRs 3 and 4, delete the private-header copy block from common-functions.sh and swap glslang's disassemble.h for the SPIRV-Tools disassembler"
    status: pending
  - id: backend-seam
    content: "PR 6: Introduce ShaderCompilerBackend interface behind ShaderManager's concrete GlslangWrapperUqPtr and add a flag-gated Slang implementation with ISlangFileSystem over Library"
    status: pending
  - id: slang-default
    content: "PR 7: Switch default to Slang, delete ShaderRedecorator and ResourceLimits, populate ShaderReflection from Slang reflection, and reimplement clash diagnostics, struct-flattening names, type whitelist, and push constant reflection"
    status: pending
  - id: shader-migration
    content: "PR 8+: Migrate shader sources to Slang modules and generics incrementally behind the PR 6 flag, retiring the string-splicing Builder"
    status: pending
isProject: false
---

# Replacing glslang with Slang and upgrading to Vulkan 1.4

## Context

The shader pipeline assembles GLSL from templates at runtime, compiles it with glslang, then post-processes the SPIR-V to assign descriptor bindings and extract reflection:

```mermaid
flowchart TD
    Library["Library (GLSL templates)"] --> Builder["ShaderManager::Builder (records textual ops)"]
    Builder --> ProcessOps["processOperators + buildExtensionAndIncludesString"]
    ProcessOps --> Glslang["GlslangWrapper::glslToSpirv"]
    Glslang --> IoMap["IoMapResolver: writes sentinel set/binding/location"]
    IoMap --> Redec["ShaderRedecorator: SPIRV-Cross reads sentinels, allocates real bindings, patches SPIR-V words"]
    Redec --> Reflect["ShaderReflection (name-keyed POD)"]
    Reflect --> Cache["ShaderCache"]
    Cache --> Material["Material / VulkanMaterial"]
    Subroutines["TShaderIRUtils: glslang AST call rebinding"] --> Glslang
```

`ShaderRedecorator` exists because glslang's I/O mapper works per-`TProgram`, and [GfxDriver/ShaderCompiler/ShaderManager.cpp](GfxDriver/ShaderCompiler/ShaderManager.cpp) calls `buildSpirv` once per stage, so glslang cannot assign bindings coherently across the stages of one material. The sentinels in [GfxDriver/ShaderCompiler/ShaderRedecorator.h](GfxDriver/ShaderCompiler/ShaderRedecorator.h) exist purely to distinguish "explicitly declared" from "needs assignment".

## The gating constraint, as resolved by PR 0a

**This section's original premise was wrong, and the correction re-sequences the plan.** It assumed the SDK was pinned at 1.3.275.0 *because* of the 15 hand-copied glslang private headers, and that PRs 3 and 4 both had to land before the SDK could move. The PR 0a spike disproved that: the tree now builds and runs correctly on SDK 1.4.363.0 with glslang 16.6.0, private headers and all. Total adaptation cost was two commits:

- `GlslangWrapper.cpp` needed `MachineIndependent/LiveTraverser.h` included ahead of `iomapper.h`, which no longer pulls in the AST types itself
- `SpirvCrossUtils.cpp` needed new `case` arms for SPIRV-Cross base types added since 1.3.275 (`CoopVecNV`, `MeshGridProperties`, `BFloat16`, `FloatE4M3`, `FloatE5M2`, `Tensor`, `DescriptorHeapBuffer`)

`FindGlslang.cmake` needed no change: the SDK still builds `MachineIndependent`, `GenericCodeGen`, and `OSDependent` as separate static libraries, so the `REQUIRED_VARS` failure the plan predicted never materialized. The "glslang changes" in the old pin comment therefore referred to something already fixed upstream, not to a standing blocker.

What this means for the rest of the plan:

- The private headers are **technical debt, not a blocker**. PRs 3 and 4 remain worth doing, because the sentinel values are derived from glslang internals and `ShaderRedecorator` is the bulk of what Slang replaces, but they no longer hold back an SDK upgrade and can be sequenced on their own merits.
- Removing the copy block from `common-functions.sh` and dropping `glslang/SPIRV/disassemble.h` becomes a cleanup pass *after* PRs 3 and 4, tracked as PR 5c, rather than a precondition for the bump.
- The bump also supplied the second thing the migration needs: `slangc` and `slang.h` are now installed in the deps prefix, so the PR 0b spike is unblocked.

For the record, the two dependencies that drive PRs 3 and 4:

- [GfxDriver/ShaderCompiler/TShaderIRUtils.cpp](GfxDriver/ShaderCompiler/TShaderIRUtils.cpp) needs `Include/InfoSink.h`, `Include/intermediate.h`, `MachineIndependent/LiveTraverser.h`, `MachineIndependent/localintermediate.h`
- [GfxDriver/ShaderCompiler/GlslangWrapper.cpp](GfxDriver/ShaderCompiler/GlslangWrapper.cpp) needs `MachineIndependent/iomapper.h` to subclass `glslang::TIoMapResolver`, which `Public/ShaderLang.h` only forward-declares, plus `Include/Types.h` transitively via `ent.symbol->getType().getQualifier()`, **and now `MachineIndependent/LiveTraverser.h` directly**, which PR 0a had to add because `iomapper.h` stopped including it

That last point is a consequence of the bump worth noting before PR 3 is scoped: `LiveTraverser.h` is no longer owned solely by `TShaderIRUtils`. Deleting `TShaderIRUtils` therefore does not retire it; only PRs 3 and 4 together do.

## The binding convention (the contract to preserve)

This is the semantic contract every PR below must keep intact. [Tests/RenderTests/ShaderCompiler/shaders/shaderRedecoratorTest.vert](Tests/RenderTests/ShaderCompiler/shaders/shaderRedecoratorTest.vert) is the canonical illustration:

```
layout(std430, binding = 0) uniform SHARED_UBO { mat4 viewTM; };   // explicit: shared across stages, keep
layout(std430) uniform VERT_UBO { float vert_x; int vert_y; };      // omitted: auto-assign, stage-local
layout(binding = 3) uniform sampler2D shared_sampler;               // explicit: shared, keep
uniform sampler2D vert_sampler;                                     // omitted: auto-assign
in vec3 in_position;                                                // no location: auto-assign
```

An explicit `binding = N` means "this resource is shared across the stages of the material, keep this number so the stages agree". An omitted binding means "allocate one for me". That distinction is the entire reason the sentinel scheme exists, and it is what `reserveBindings` (reserve the explicit ones first) and `allocateBindings` (fill the gaps) implement.

## Verified facts that shape the plan

- No shader declares an explicit `set =` anywhere, so all set-handling branches in `ShaderRedecorator` guard a case that never occurs, consistent with the hardcoded `int set = 0` in `redecorateInternal`. Only the binding distinction is real. Re-verified during PR 1 against all 161 shader sources, and now enforced rather than merely observed: `ShaderRedecoratorTest` asserts every reflected resource reports set 0.
- Vertex attribute locations are reassigned unconditionally; the location sentinel is never tested, only the presence of a `Location` decoration matters. This makes `kUninitializedLocation` far easier to eliminate than the binding sentinel.
- The sentinel values are "one less than `glslang::TQualifier::layout*End`" per the comment in `ShaderRedecorator.h:22-29`, that is, derived from glslang's internal "undefined" markers. They are therefore inherently version-fragile. PR 0a showed the values happened to survive 1.3.275 to 16.6.0 unchanged, so this is a latent hazard rather than an active one, but nothing guarantees the next bump is as kind.
- `replaceFunctionCall` has no live callers; it is only reachable via builder deserialization at `ShaderManager.cpp:354`. The only live producer of the rebind map is `addSubroutineBinding`, keyed on bare function names
- The AST path does not support function overloads (`TShaderIRUtils.cpp:57-59`), so textual substitution gives up no capability
- Includes are resolved textually by `buildExtensionAndIncludesString` and prepended *after* `processOperators` runs, so any textual subroutine pass must be a post-assembly phase over `final_glsl`, not an `OpType`
- [GfxDriver/ShaderCompiler/ResourceLimits.cpp](GfxDriver/ShaderCompiler/ResourceLimits.cpp) is a vendored copy of glslang's `StandAlone/ResourceLimits.cpp`, so `DefaultTBuiltInResource` is already local
- The shader library declares its uniform blocks `layout(std430)`, which is only legal because `uniformBufferStandardLayout` and `scalarBlockLayout` are **hard device requirements**: `VulkanPlatform::isDeviceSuitable` rejects a device lacking either, and `createDeviceContext` enables both unconditionally. Surfaced by PR 1, where `spirv-val` rejected `pointTemplate.vert` under its default extended-layout rules (`Array stride 8 must satisfy alignment 16` on a `double[2]` in a UBO). Any replacement backend has to reproduce these layout rules, so this is a constraint on PR 7, where Slang needs the equivalent of `-fvk-use-scalar-layout` or per-block layout attributes rather than its defaults.

## Inventory

Recounted, because the original figures were internally inconsistent ("roughly 130" against a 122 + 24 breakdown that sums to 146):

- **161** shader sources: 137 stage files plus 24 shared `.glsl` include files. By extension: 50 `.frag`, 42 `.vert`, 24 `.glsl`, 18 `.comp`, 6 `.mesh`, 5 each of `.miss`, `.geom` and `.closest`, 4 `.raygen`, 1 each of `.task` and `.isect`.
- **105** of them declare uniform or buffer resources, totalling **274** declarations.

The split by location matters more than the totals, and was missing before: **85 of the 161 live under `Tests/`**, against 52 in `QueryRenderer`, 22 in `GfxDriver` and 2 in `QueryEngine`. So more than half the corpus is test fixtures.

That rescopes two things. PR 4's fallback approach touches **55 files and 162 declarations** outside `Tests/`, not the ~300 originally quoted, which makes the mechanical option roughly half as expensive as feared. And the PR 0b spike should report test and non-test failures separately, since a test fixture that Slang rejects can be rewritten or deleted freely, while a production shader cannot.

## The reflection contract

`ShaderReflection` is the seam to preserve throughout. Any replacement backend must be able to populate all of it.

Corrected count: 29 accessor call sites, not 21, and they span two files rather than one. [GfxDriver/Pipeline/Material.cpp](GfxDriver/Pipeline/Material.cpp) has 13 and [GfxDriver/Drivers/Vulkan/Pipeline/VulkanMaterial.cpp](GfxDriver/Drivers/Vulkan/Pipeline/VulkanMaterial.cpp) has 16. The earlier list also omitted acceleration structures entirely, which is the kind of gap that produces a Slang backend that looks complete until a raytracing shader is loaded. The full consumed surface:

- `getAllUniformBufferNames`, `getUniformBufferBinding`, `getUniformBufferBlockSize` (used to size and create local UBOs)
- `hasVertexAttr`, `getVertexAttrLocation`
- `hasUniformBufferAttr`, `getUniformBufferAttrItemInfo`, `getShaderStorageBufferAttrItemInfo`
- `getAllSamplerNames`, `getSamplerBinding`, `getSamplerArraySize`
- `getAllStorageImageNames`, `getStorageImageBinding`, `getStorageImageArraySize`
- `getAllShaderStorageBufferNames`, `getShaderStorageBufferBinding`
- `getAllAccelerationStructureNames`, `getAccelerationStructureBinding`, `getAccelerationStructureSize`
- `hasFragmentShaderOutputLocation`, consumed by `VulkanGraphicsPipeline.cpp:133` to decide blend attachment state

`ShaderManager::buildSpirv` additionally uses `getUniformBufferBinding` to validate that every name passed to `setExternalUniformBuffers` actually exists in the shader (`ShaderManager.cpp:1043-1047`), and `createCacheVector` uses `getAllUniformBufferAttrNames` / `getAllShaderStorageBufferAttrNames` with `ItemInfo` equality to reject duplicate non-shared attribute names across stages. Both checks must keep working.

PR 1 pins all of this, so a backend swap that drops part of the surface now fails a test rather than a render.

`ShaderReflection` has no push-constant support; those are hand-maintained separately in [GfxDriver/Pipeline/PushConstantRanges.cpp](GfxDriver/Pipeline/PushConstantRanges.cpp).

## PR 0 - Spikes (no production code)

Two cheap experiments that decide branches below.

**0a, done.** The SDK bump, executed per [.cursor/plans/vulkan-sdk-1.4.363-fork.md](.cursor/plans/vulkan-sdk-1.4.363-fork.md) and landed rather than discarded, because it turned out to cost two small commits instead of the API-drift slog the plan budgeted for. See the gating-constraint section above for the findings and their consequences. The surviving pieces of what was PR 5 are now PR 5b and PR 5c.

**0b, next.** Compile all shader sources through `slangc`'s GLSL compatibility mode (`-allow-glsl`) and count failures. This sizes PR 6 onward and determines whether incremental adoption is viable. `slangc` now ships in the deps prefix at `$PREFIX/bin/slangc`, installed by the `slang` target added to the `install_vulkan` invocation.

The spike needs to distinguish three outcomes per file, because only the first is cheap:

- compiles clean under `-allow-glsl`
- fails on something mechanical and pattern-fixable across many files
- fails on something Slang's GLSL subset genuinely does not model, which forces that shader into the PR 8 rewrite bucket early

Note that stage files are not independently compilable: they depend on the `#include` dictionary assembled at runtime by `buildExtensionAndIncludesString`, and on the `processOperators` textual substitutions. A spike that feeds raw template sources to `slangc` will report failures that are artifacts of missing includes rather than real Slang gaps. Either run the spike over assembled `final_glsl` captured from a real run (the `ShaderArtifactTypeBits` hooks can emit it), or accept that the raw-source pass only produces a lower bound and triage accordingly.

## PR 1 - Replace byte-exact SPIR-V goldens with semantic assertions

**Landed as 95fb438ec0.** Had to land first, because every later PR changes SPIR-V output.

[Tests/RenderTests/ShaderCompiler/ShaderCompilerTest.cpp](Tests/RenderTests/ShaderCompiler/ShaderCompilerTest.cpp) previously asserted:

```
    EXPECT_EQ(read_spirv, spirv_gen);
```

The README already documents regenerating goldens whenever the toolchain moves, which makes the test vacuous exactly when it is needed. Replace with assertions on full `ShaderReflection` contents (set, binding, location, offset, size, array size, keyed by name), `spirv-val` on each blob, and the existing image-comparison render tests as backstop. Extend `ShaderRedecoratorTest` and `ShaderRedecoratorClashTest`, which are already the right shape. Add a reflection-dump artifact using the existing `ShaderArtifactTypeBits::kSpvReflect` hook.

Corrected: the `.spv` goldens *are* checked in, 18 of them across `Manual/Results` and `Renderer/Results`. The earlier claim that only `Inputs/*.builder` was present was wrong.

PR 0a then produced a live demonstration of why this test needs replacing. Moving to glslang 16.6.0 required regenerating 9 of the goldens, one of which changed size (`SMAANeighborhoodBlending.frag.spv`, 6896 to 6832 bytes). A size change means codegen genuinely changed, and the test offered no way to tell whether the new output was semantically equivalent; the only available response was to accept the new bytes. That commit is the argument for this PR: cite it in review.

### What was built

The blocker, found on implementation: the parameterized `FromFile` tests ran through `createCache`, which passes `nullptr` as the redecorator (`ShaderManager.cpp:1263`), and `buildSpirv` only populates reflection when that pointer is non-null (`ShaderManager.cpp:1036-1040`). Every one of those caches therefore carried an **empty** `ShaderReflection`. Asserting on it as the plan originally prescribed would have produced a test that passes unconditionally while looking more rigorous than the byte comparison it replaced. Resolved by routing those tests through `createCacheVector` with a one-element vector, which is the production entry point and does run the redecorator.

- `ShaderReflection::serialize` writes a name-sorted text form of all eight maps plus the fragment output locations, printing every `ItemInfo` field including the ones a category leaves at -1. The 18 `Results/*.spv` files are replaced by 9 `Results/*.reflect` files in that format, one per live test case, and `--regenerate-spv` becomes `--regenerate-reflect`.
- `spirv-val` runs on every generated blob, with the target environment derived from the blob's own declared SPIR-V version so that PR 5b does not need to edit the test in lockstep.
- The validator is configured with `SetUniformBufferStandardLayout` and `SetScalarBlockLayout` to match the device features the renderer requires. See the layout entry under Verified facts; without this the validator is stricter than any device we run on.
- Failures report the first differing line rather than two whole documents, since a diffable golden is the entire point.
- `ShaderRedecoratorTest` keeps its explicit binding table and gains assertions for descriptor sets, array sizes, block sizes, UBO member offsets, SSBO member naming, vertex attribute locations, fragment output locations, and resource-list completeness. It now also asserts the binding convention directly: shared resources must agree across stages, stage-local ones must not collide.
- `ShaderRedecoratorClashTest` pins the clash message content, not just the throw, because PRs 4 and 7 replace the allocator that produces it.

Not tests only, contrary to the original note, and the deviation is deliberate: `serialize` lives in production so the `kSpvReflect` artifact can emit the same format the goldens use, which is what makes an artifact from a failing run directly diffable against a golden. That also required a const `ShaderCache::getReflection` overload and a new `write_spirv_artifacts` parameter.

### What it caught immediately

`spirv-val` rejected `pointTemplate.vert`, which has a `double[2]` in a uniform block with stride 8. That shader had never been validated at all, despite its golden being regenerated through at least one toolchain bump. It is legal only under the relaxed layout rules the renderer requires, which is what produced the layout entry under Verified facts. Vindicates the approach on the first run.

Reviewing the generated baseline also turned up the reflection array-dimension gap now recorded under PR 7.

### Residual risk

- Reflection covers the interface, not function bodies. A shader that computes something different with an unchanged interface will not be caught here. That is the intended trade, with `spirv-val` and the image-comparison render tests as backstop, but it is a real reduction in coverage and was called out as such in the commit message rather than sold as a pure win.
- Seven of the deleted `.spv` goldens had no live test case and were dropped without replacement: the four `fastSymbolTemplate` files and the three `lineTemplate` files, matching cases commented out in `builder_to_spirv_legacy_cases`, plus the orphaned `MSMultiGpuComposite.frag.spv`. Re-enabling any of those cases now needs a fresh `--regenerate-reflect` rather than a baseline from history.
- Retired: routing `FromFile` through `createCacheVector` means those shaders are now redecorated during the test, which raised the possibility that one would trip `validate_buffer_attr_type` or the "nested structs not supported" check. None did; all 9 cases pass.

## PR 2 - Hygiene and dead code

No behaviour change; makes later diffs readable.

- `kShaderStageCount = 8` in [GfxDriver/ShaderCompiler/Types.h](GfxDriver/ShaderCompiler/Types.h) is marked "must be kept in sync with enum" but `ShaderStage` has 14 entries. Currently unreferenced, so latent.
- `find_error_line_number` in `GlslangWrapper.cpp:213-235` scrapes glslang's log format and `CHECK_NE`s on it, hard-crashing debug builds if the format changes, and reports only the first error.
- `ShaderRedecorator::getReservedBindings` ignores its `resource_type` parameter; `redecorateInternal` is always called with `set = 0`. Delete the vestigial multi-set scaffolding (including `ShaderReflection::validateSet`, which only checks `!= -1`) or document why it stays.
- `ShaderReflection::clear` does not clear `fragment_shader_output_locations_`, and `ShaderReflection::operator=` does not copy it. Both are latent rather than live: `clear` is only reached via `initialize()` on a freshly constructed object in `redecorate`, and `operator=` has no callers. The field itself is live, though, since `VulkanGraphicsPipeline.cpp:133` uses it to decide blend attachment state, so fix both rather than leaving the trap armed. Found while writing PR 1.
- `ShaderReflection` declares a destructor and a copy-assignment operator, which suppresses its implicit move constructor, so the `std::move(reflection)` into the `ShaderCache` constructor at `ShaderManager.cpp:1125` silently copies. Harmless today, and the copy is what currently saves the `operator=` bug above from being live, but the two interact and should be fixed together.

## PR 3 - Remove TShaderIRUtils, resolve subroutines textually

Retires 3 of the 15 private headers: `Include/InfoSink.h`, `Include/intermediate.h` and `MachineIndependent/localintermediate.h`. Note it does **not** retire `MachineIndependent/LiveTraverser.h`, despite that being listed against `TShaderIRUtils` above, because PR 0a had to add a direct include of it to `GlslangWrapper.cpp`. Verify `iomapper.h` does not pull `localintermediate.h` in transitively before assuming even those three go.

Delete [GfxDriver/ShaderCompiler/TShaderIRUtils.cpp](GfxDriver/ShaderCompiler/TShaderIRUtils.cpp) and its header, and drop the `rebind_tshader_function_calls` call from `glslToSpirv`.

Implement subroutine resolution as a post-assembly textual pass over `final_glsl`, after includes are prepended. It cannot be an `OpType` because targets such as `transformRGBtoRGB` live in `colorConvertSubroutines.glsl`, which is not present during `processOperators`.

Requirements:
- Match on `name(` and handle the source function's own definition explicitly. The AST path currently relies on glslang's call-graph dead-code elimination to drop what becomes unreachable.
- `is_required` needs lexical detection of the target definition; reuse the existing `get_function_bounds` helper.
- Tolerate transient empty targets: `addSubroutineBinding("getAccumulatedColor", "", true)` at `QueryRenderer/Scales/ScaleAccumState.cpp:526` is overwritten at line 575.

Residual risk to note in review: targets reaching the shader via a literal `#include` resolved by `GlslangIncluder` rather than the include dictionary are invisible to text-time inspection.

## PR 4 - Remove IoMapResolver, drop iomapper.h

No longer "the other half of the unpin", since PR 0a unpinned the SDK without either PR 3 or PR 4. What PR 4 still buys is deleting the sentinel scheme and most of `ShaderRedecorator`, which is the bulk of what Slang replaces in PR 7, plus the last of the private headers.

Primary approach: link all stages of a material into a single `glslang::TProgram` and let glslang's own default resolver assign coherent cross-stage bindings. `TProgram::mapIO(nullptr, nullptr)` is public API, so no private header and no shader source churn. `TDefaultGlslIoResolver` keys its slot map by name across stages, which is exactly the sharing semantics the sentinels currently emulate.

Changes:
- Restructure `createCacheVector` and `buildSpirv` from per-stage to per-material: run `processOperators` for all builders, create all `TShader`s, add all to one `TProgram`, link and `mapIO` once, then call `GlslangToSpv` per stage intermediate (already the shape, via `program->getIntermediate(stage)`).
- Delete the `IoMapResolver` class, the three sentinel constants, and `ShaderRedecorator`'s allocation logic. `ShaderRedecorator` reduces to reflection extraction via SPIRV-Cross.
- `createCache` (single builder, explicitly does not redecorate) stays a single-stage `TProgram`. It now has exactly one live caller, `saveArtifacts` at `ShaderManager.cpp:1159`, and after PR 1 moved the `FromFile` tests onto `createCacheVector` no test exercises it directly. Treat changes to it as unverified by the suite.
- Preserve equivalents for `reserveBindings` clash diagnostics (covered by `ShaderRedecoratorClashTest`) and the `kMaxBindingsPerSet` overflow checks, since glslang's resolver will not produce the same messages.

Bindings will differ from today, and PR 1's goldens record absolute set and binding numbers, so expect every `.reflect` file to need regeneration. That is the mechanism working as intended rather than a problem: the diff is the review artifact, and what matters is that names, block sizes and array sizes stay put while only the numbers move. All runtime consumers are reflection-driven, so no consumer should need changing.

Fallback if the single-`TProgram` approach misbehaves: declare the sentinels explicitly in shader source behind macros (`layout(set = AUTO_SET, binding = AUTO_BINDING)`), keeping `ShaderRedecorator` byte-identical. This is mechanical and, per the recounted inventory, touches 162 declarations across 55 files outside `Tests/` (274 across 105 including them), about half the originally quoted size. It still makes every resource declaration noisier and requires confirming glslang accepts out-of-range values as explicit source qualifiers.

Rejected alternative, recorded so nobody spends a day on it: using glslang's public `setShiftUboBinding` / `setShiftSsboBinding` / `setShiftSamplerBinding` / `setShiftImageBinding` to push auto-assigned bindings into a high range, so the auto-versus-explicit test becomes a threshold comparison instead of a sentinel match. This looks ideal because it needs no private header and no shader changes, but it does not work: `TDefaultIoResolver::resolveBinding` adds the shift base to *explicitly declared* bindings as well as auto-assigned ones, so both land in the shifted range and remain indistinguishable. The shift is an HLSL register-offset feature, not an auto-allocation base.

## PR 5a - Vulkan SDK 1.4.363.0 with Slang (done, landed early)

The dependency tree now builds `loader glslang spirvcross vul layers slang` from a fork of the upstream 1.4.363.0 script. No longer gated on PRs 3 and 4, which is the plan's main correction. Details in [.cursor/plans/vulkan-sdk-1.4.363-fork.md](.cursor/plans/vulkan-sdk-1.4.363-fork.md).

Two loose ends it deliberately left:

- Slang attribution and OSRB approval, Task 3 of that plan, still outstanding. Needed before anything ships with Slang linked in, so it gates PR 6's merge rather than its development.
- `ThirdParty/vulkan/vulkansdk-1.3.275.0` was deleted, so reverting the bump is no longer a one-line `VULKAN_VERSION` change. Revert the commit instead.

## PR 5b - Retarget the compiler to SPIR-V 1.6, still on glslang

Independent of PRs 3 and 4. Now that the SDK supports it, this is the remaining version work, and it is the risk-bearing half: PR 5a changed which toolchain compiles the shaders, while this changes what the toolchain is asked to emit and what the driver advertises.

- Resolve the existing mismatch: the instance requests `VK_API_VERSION_1_3` (`GfxDriver/Drivers/Vulkan/VulkanPlatform.cpp:39`) while the compiler targets `EShTargetVulkan_1_2` / `EShTargetSpv_1_4` (`GlslangWrapper.cpp:274-275`). Not a correctness bug, just capability left on the table: targeting lower is conservative and runs fine on a 1.3 device.
- Separate the two halves, because only one is free. The **compiler** retarget to `EShTargetVulkan_1_3` / `EShTargetSpv_1_6` costs nothing: `VULKAN_API_VERSION` doubles as a hard minimum device version (`VulkanPlatform.cpp:514-515` rejects anything lower), so the device floor is already Vulkan 1.3, and SPIR-V 1.6 is core in 1.3. The **instance** bump to 1.4 is a deployment decision: it raises the NVIDIA driver floor from the currently enforced 535 (`VulkanPlatform.cpp:556`) to whatever first reported Vulkan 1.4, around R570, and buys the shader compiler nothing because Vulkan 1.4 also tops out at SPIR-V 1.6. Confirm the supported-driver matrix before taking it, and keep it revertible on its own.
- Adopt what 1.4 promotes out of the optional capability extension handling in `VulkanPlatform.cpp:66-94`. Checked: **nothing to do here.** Every extension the driver requests (external memory and semaphore FD, swapchain, mesh shader, fragment shading rate, fragment shader interlock, ray tracing pipeline, acceleration structure, deferred host operations, ray query, memory budget, debug utils, validation features) stays an extension in 1.4. What 1.4 promotes, such as push descriptor, maintenance5, map memory2, index type uint8 and host image copy, this codebase does not use. Keep the bullet only as a record that it was checked.
- Normalize `GfxDriver/Drivers/Vulkan/VulkanPhysicalDevice.cpp:84-85`, where "is at least 1.2" is spelled `major > 1 || minor > 1`.

Recommended split: take the compiler retarget and the comparison fix as PR 5b, and leave the instance bump to 1.4 as a separate PR to be scheduled against the driver matrix. Doing so means the plan's Vulkan 1.4 goal is reached in the toolchain immediately and in the instance only when deployment allows.

## PR 5c - Drop the private headers

Pure cleanup, unblocked only once PRs 3 and 4 have removed the last consumers. Tracked separately so the debt does not get forgotten now that it no longer blocks anything.

- Delete the header-copying block and the `mkdir -p` guards at `scripts/common-functions.sh:1091-1111`, and the accompanying comment at 1077-1080.
- Replace `glslang/SPIRV/disassemble.h` in [GfxDriver/ShaderCompiler/SpirvArtifacts.cpp](GfxDriver/ShaderCompiler/SpirvArtifacts.cpp) line 11 with the SPIRV-Tools disassembler. This one is already independent of PRs 3 and 4 and could ride along with either.

Verify by confirming a deps build with the block removed still produces a tree the renderer compiles against; the headers are copied into the prefix, so a stale prefix will mask the failure.

## PR 6 - Introduce a compiler backend seam, add Slang alongside glslang

Purely additive. [GfxDriver/ShaderCompiler/ShaderManager.h](GfxDriver/ShaderCompiler/ShaderManager.h) line 315 holds a concrete `GlslangWrapperUqPtr`; introduce a `ShaderCompilerBackend` interface (compile plus reflect) and add a Slang implementation selectable by flag, with glslang remaining the default. Map `GlslangIncluder` onto a Slang `ISlangFileSystem` over `Library`. Nothing changes for existing callers.

Scope depends on spike 0b. Also needs Slang OSRB approval before it can merge, carried over from PR 5a.

## PR 7 - Switch the default to Slang, delete ShaderRedecorator

Slang assigns descriptor sets and bindings across a composed program with multiple entry points and reports them through reflection, so the remaining redecoration, SPIRV-Cross patching via `get_binary_offset_for_decoration`, and `ResourceLimits` / `TBuiltInResource` all go away rather than being ported. This also resolves the never-completed TODO at `GlslangWrapper.cpp:201` about sourcing limits from the device, since limit validation moves to the driver.

Keep `ShaderReflection` unchanged and populate it from Slang reflection. Four behaviours need deliberate reimplementation:

- Clash diagnostics quality from `reserveBindings` / `allocateBindings`
- The struct-flattening naming convention at `ShaderRedecorator.cpp:306-355`: dotted names for nested UBO members, bare names for SSBO members, depended on verbatim by `Material` and the Vega property writers
- `validate_buffer_attr_type`, which whitelists only scalar int/uint/int64/uint64/float/double; decide whether to carry the restriction forward
- `ShaderReflection` cannot express an array dimension on a buffer member, and silently drops it. `SLAB_ADDRESS_TABLE_UBO` declares `SlabAddressTableEntry slabs[64]` ([QueryRenderer/Marks/shaders/slabAddressTable.glsl](QueryRenderer/Marks/shaders/slabAddressTable.glsl)), and the reflection records a `block_size` of 1024 alongside exactly two attrs, `slabs.cuda` at offset 0 and `slabs.vulkan` at offset 8, as though the array were a single element. Harmless today only because that block is bound whole via `bindExternalUniformBufferToBlock`, so nothing consumes the per-member offsets. Slang reflection will report the array properly, so decide deliberately whether to represent it or keep discarding it; either way the PR 1 golden for `pointTemplate.vert` changes, and that diff is expected rather than a regression. The same gap applies to `uDomains_x_x`, a `double[2]` reported as one 16-byte attr.
- Push constants, currently hand-maintained in [GfxDriver/Pipeline/PushConstantRanges.cpp](GfxDriver/Pipeline/PushConstantRanges.cpp) and absent from reflection, come free here

## PR 8+ - Migrate shader sources to Slang modules and generics

Largest effort, biggest long-term payoff, safely deferrable. Slang interfaces plus generics with link-time specialization replace both the subroutine mechanism from PR 3 and the string-splicing `Builder` itself, which is the root source of the complexity this subsystem exists to manage. Migrate per-shader or per-family behind the PR 6 flag rather than as a flag day.

## Open risks

- Whether glslang's default GLSL I/O resolver assigns bindings coherently across stages in a single `TProgram` while respecting explicit bindings. Gates PR 4's primary approach; the fallback is documented above.
- How many of the 161 shader sources clear Slang's GLSL subset. Gates PR 6 onward, and is what spike 0b measures. Only the 76 outside `Tests/` are genuinely load-bearing, so the spike should split its counts that way.
- Slang OSRB approval. Not a technical risk but a hard gate on shipping anything from PR 6 onward, and entirely outside this plan's control, so start it now rather than when the code is ready.

Retired: the SDK bump itself, which PR 0a landed.

PRs 1 through 3 are worth doing regardless of how spike 0b resolves.
