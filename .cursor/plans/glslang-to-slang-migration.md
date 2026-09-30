---
name: glslang to slang migration
overview: Replace glslang with Slang and upgrade to Vulkan 1.4, sequenced as eight independently shippable PRs that each preserve full functionality, starting with a test safety net and ending with optional shader source migration.
todos:
  - id: spikes
    content: "PR 0: Spike the SDK bump on a scratch branch to collect actual errors, and compile all shader sources through Slang's GLSL compatibility mode to count failures"
    status: pending
  - id: test-safety-net
    content: "PR 1: Replace byte-exact SPIR-V golden comparison in ShaderCompilerTest.cpp with ShaderReflection equality assertions plus spirv-val; add reflection-dump artifact via existing kSpvReflect hook"
    status: pending
  - id: hygiene
    content: "PR 2: Fix kShaderStageCount, harden glslang error-log scraping, remove vestigial multi-set scaffolding in ShaderRedecorator and ShaderReflection::validateSet"
    status: pending
  - id: remove-tshaderirutils
    content: "PR 3: Delete TShaderIRUtils and reimplement subroutine binding as a post-assembly textual pass over final_glsl, after includes are prepended"
    status: pending
  - id: remove-iomapresolver
    content: "PR 4: Link all stages of a material into a single glslang::TProgram using public mapIO(nullptr, nullptr); delete IoMapResolver, sentinels, and ShaderRedecorator's allocation logic, preserving clash diagnostics"
    status: pending
  - id: vulkan-14
    content: "PR 5: Fix FindGlslang for consolidated library, swap disassemble.h for SPIRV-Tools, remove private-header copying from common-functions.sh, and align instance and compiler targets to Vulkan 1.4 / SPIR-V 1.6"
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

## The gating constraint

The Vulkan SDK is pinned at 1.3.275.0 by [scripts/common-functions.sh](scripts/common-functions.sh), which hand-copies 15 non-installed glslang private headers. Two independent dependencies cause this, and **both** must go before the SDK can move:

- [GfxDriver/ShaderCompiler/TShaderIRUtils.cpp](GfxDriver/ShaderCompiler/TShaderIRUtils.cpp) needs `Include/InfoSink.h`, `Include/intermediate.h`, `MachineIndependent/LiveTraverser.h`, `MachineIndependent/localintermediate.h`
- [GfxDriver/ShaderCompiler/GlslangWrapper.cpp](GfxDriver/ShaderCompiler/GlslangWrapper.cpp) line 11 needs `MachineIndependent/iomapper.h` to subclass `glslang::TIoMapResolver`, which `Public/ShaderLang.h` only forward-declares, plus `Include/Types.h` transitively via `ent.symbol->getType().getQualifier()`

Removing either alone does not unpin the SDK. PRs 3 and 4 together do.

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

- No shader declares an explicit `set =` anywhere, so all set-handling branches in `ShaderRedecorator` guard a case that never occurs, consistent with the hardcoded `int set = 0` in `redecorateInternal`. Only the binding distinction is real.
- Vertex attribute locations are reassigned unconditionally; the location sentinel is never tested, only the presence of a `Location` decoration matters. This makes `kUninitializedLocation` far easier to eliminate than the binding sentinel.
- The sentinel values are "one less than `glslang::TQualifier::layout*End`" per the comment in `ShaderRedecorator.h:22-29`, that is, derived from glslang's internal "undefined" markers. They are therefore inherently version-fragile, which is part of why the SDK is pinned.
- `replaceFunctionCall` has no live callers; it is only reachable via builder deserialization at `ShaderManager.cpp:354`. The only live producer of the rebind map is `addSubroutineBinding`, keyed on bare function names
- The AST path does not support function overloads (`TShaderIRUtils.cpp:57-59`), so textual substitution gives up no capability
- Includes are resolved textually by `buildExtensionAndIncludesString` and prepended *after* `processOperators` runs, so any textual subroutine pass must be a post-assembly phase over `final_glsl`, not an `OpType`
- [GfxDriver/ShaderCompiler/ResourceLimits.cpp](GfxDriver/ShaderCompiler/ResourceLimits.cpp) is a vendored copy of glslang's `StandAlone/ResourceLimits.cpp`, so `DefaultTBuiltInResource` is already local

## Inventory

- Roughly 130 shader sources: 122 stage files (`.vert`, `.frag`, `.comp`, `.geom`, `.mesh`, `.task`, `.raygen`, `.closest`, `.isect`, and similar) plus 24 shared `.glsl` include files
- Around 108 of those declare uniform or buffer resources, totalling roughly 300 declarations. This is the sizing for PR 4's fallback approach.

## The reflection contract

`ShaderReflection` is the seam to preserve throughout; `Material` depends on it in 21 places. Any replacement backend must be able to populate all of it. The accessor surface actually consumed by [GfxDriver/Pipeline/Material.cpp](GfxDriver/Pipeline/Material.cpp) is:

- `getAllUniformBufferNames`, `getUniformBufferBinding`, `getUniformBufferBlockSize` (used to size and create local UBOs)
- `hasVertexAttr`, `getVertexAttrLocation`
- `hasUniformBufferAttr`, `getUniformBufferAttrItemInfo`, `getShaderStorageBufferAttrItemInfo`
- `getSamplerBinding` / `getSamplerArraySize`, `getStorageImageBinding` / `getStorageImageArraySize`

`ShaderManager::buildSpirv` additionally uses `getUniformBufferBinding` to validate that every name passed to `setExternalUniformBuffers` actually exists in the shader (`ShaderManager.cpp:1043-1047`), and `createCacheVector` uses `getAllUniformBufferAttrNames` / `getAllShaderStorageBufferAttrNames` with `ItemInfo` equality to reject duplicate non-shared attribute names across stages. Both checks must keep working.

`ShaderReflection` has no push-constant support; those are hand-maintained separately in [GfxDriver/Pipeline/PushConstantRanges.cpp](GfxDriver/Pipeline/PushConstantRanges.cpp).

## PR 0 - Spikes (no production code)

Two cheap experiments that decide branches below. Time-box each to a day.

1. Attempt the SDK bump on a scratch branch and collect the actual compiler and CMake errors. This establishes how much is plumbing versus real API drift. Expect at minimum that [cmake/Modules/FindGlslang.cmake](cmake/Modules/FindGlslang.cmake) fails immediately: its `REQUIRED_VARS` hard-requires `MachineIndependent`, `GenericCodeGen`, and `OSDependent` as separate libraries, which glslang 14 consolidated into a single `glslang` target.
2. Compile all shader sources through Slang's GLSL compatibility mode (`-allow-glsl`) and count failures. This costs PR 6 onward and determines whether incremental adoption is viable.

## PR 1 - Replace byte-exact SPIR-V goldens with semantic assertions

Must land first; every later PR changes SPIR-V output.

[Tests/RenderTests/ShaderCompiler/ShaderCompilerTest.cpp](Tests/RenderTests/ShaderCompiler/ShaderCompilerTest.cpp) currently asserts:

```
    EXPECT_EQ(read_spirv, spirv_gen);
```

The README already documents regenerating goldens whenever the toolchain moves, which makes the test vacuous exactly when it is needed. Replace with assertions on full `ShaderReflection` contents (set, binding, location, offset, size, array size, keyed by name), `spirv-val` on each blob, and the existing image-comparison render tests as backstop. Extend `ShaderRedecoratorTest` and `ShaderRedecoratorClashTest`, which are already the right shape. Add a reflection-dump artifact using the existing `ShaderArtifactTypeBits::kSpvReflect` hook.

Note that the `Manual/Results` and `Renderer/Results` directories holding the `.spv` goldens are not present in the repository; only the `Inputs/*.builder` files are checked in. Confirm where the goldens actually live before assuming there is anything to migrate from, and check whether the affected tests are currently passing, skipped, or generating their own baselines via `--regenerate-spv`.

Tests only; no production change.

## PR 2 - Hygiene and dead code

No behaviour change; makes later diffs readable.

- `kShaderStageCount = 8` in [GfxDriver/ShaderCompiler/Types.h](GfxDriver/ShaderCompiler/Types.h) is marked "must be kept in sync with enum" but `ShaderStage` has 14 entries. Currently unreferenced, so latent.
- `find_error_line_number` in `GlslangWrapper.cpp:210-232` scrapes glslang's log format and `CHECK_NE`s on it, hard-crashing debug builds if the format changes, and reports only the first error.
- `ShaderRedecorator::getReservedBindings` ignores its `resource_type` parameter; `redecorateInternal` is always called with `set = 0`. Delete the vestigial multi-set scaffolding (including `ShaderReflection::validateSet`, which only checks `!= -1`) or document why it stays.

## PR 3 - Remove TShaderIRUtils, resolve subroutines textually

Retires 3 of the 15 private headers. Delete [GfxDriver/ShaderCompiler/TShaderIRUtils.cpp](GfxDriver/ShaderCompiler/TShaderIRUtils.cpp) and its header, and drop the `rebind_tshader_function_calls` call from `glslToSpirv`.

Implement subroutine resolution as a post-assembly textual pass over `final_glsl`, after includes are prepended. It cannot be an `OpType` because targets such as `transformRGBtoRGB` live in `colorConvertSubroutines.glsl`, which is not present during `processOperators`.

Requirements:
- Match on `name(` and handle the source function's own definition explicitly. The AST path currently relies on glslang's call-graph dead-code elimination to drop what becomes unreachable.
- `is_required` needs lexical detection of the target definition; reuse the existing `get_function_bounds` helper.
- Tolerate transient empty targets: `addSubroutineBinding("getAccumulatedColor", "", true)` at `QueryRenderer/Scales/ScaleAccumState.cpp:526` is overwritten at line 575.

Residual risk to note in review: targets reaching the shader via a literal `#include` resolved by `GlslangIncluder` rather than the include dictionary are invisible to text-time inspection.

## PR 4 - Remove IoMapResolver, drop iomapper.h

The other half of the unpin. Primary approach: link all stages of a material into a single `glslang::TProgram` and let glslang's own default resolver assign coherent cross-stage bindings. `TProgram::mapIO(nullptr, nullptr)` is public API, so no private header and no shader source churn. `TDefaultGlslIoResolver` keys its slot map by name across stages, which is exactly the sharing semantics the sentinels currently emulate.

Changes:
- Restructure `createCacheVector` and `buildSpirv` from per-stage to per-material: run `processOperators` for all builders, create all `TShader`s, add all to one `TProgram`, link and `mapIO` once, then call `GlslangToSpv` per stage intermediate (already the shape, via `program->getIntermediate(stage)`).
- Delete the `IoMapResolver` class, the three sentinel constants, and `ShaderRedecorator`'s allocation logic. `ShaderRedecorator` reduces to reflection extraction via SPIRV-Cross.
- `createCache` (single builder, explicitly does not redecorate) stays a single-stage `TProgram`.
- Preserve equivalents for `reserveBindings` clash diagnostics (covered by `ShaderRedecoratorClashTest`) and the `kMaxBindingsPerSet` overflow checks, since glslang's resolver will not produce the same messages.

Bindings will differ from today. All consumers are reflection-driven, and PR 1's tests verify this.

Fallback if the single-`TProgram` approach misbehaves: declare the sentinels explicitly in shader source behind macros (`layout(set = AUTO_SET, binding = AUTO_BINDING)`), keeping `ShaderRedecorator` byte-identical. This is mechanical but touches roughly 300 declarations across 108 files, makes every resource declaration noisier, and requires confirming glslang accepts out-of-range values as explicit source qualifiers.

Rejected alternative, recorded so nobody spends a day on it: using glslang's public `setShiftUboBinding` / `setShiftSsboBinding` / `setShiftSamplerBinding` / `setShiftImageBinding` to push auto-assigned bindings into a high range, so the auto-versus-explicit test becomes a threshold comparison instead of a sentinel match. This looks ideal because it needs no private header and no shader changes, but it does not work: `TDefaultIoResolver::resolveBinding` adds the shift base to *explicitly declared* bindings as well as auto-assigned ones, so both land in the shifted range and remain indistinguishable. The shift is an HLSL register-offset feature, not an auto-allocation base.

## PR 5 - Vulkan SDK 1.4 and SPIR-V 1.6, still on glslang

Unblocked once PRs 3 and 4 land. Consider splitting the SDK bump from the version retarget, as they are independent risk sources.

- Fix `FindGlslang.cmake` for the consolidated `glslang` library target.
- Replace `glslang/SPIRV/disassemble.h` in [GfxDriver/ShaderCompiler/SpirvArtifacts.cpp](GfxDriver/ShaderCompiler/SpirvArtifacts.cpp) with the SPIRV-Tools disassembler.
- Delete the header-copying block and version pin comment at `scripts/common-functions.sh:1076-1105`.
- Resolve the existing mismatch: the instance requests `VK_API_VERSION_1_3` (`GfxDriver/Drivers/Vulkan/VulkanPlatform.cpp:39`) while the compiler targets `EShTargetVulkan_1_2` / `EShTargetSpv_1_4` (`GlslangWrapper.cpp:271-272`). Align both to Vulkan 1.4 and SPIR-V 1.6.
- Adopt what 1.4 promotes out of the optional extension handling in `VulkanPlatform.cpp:59-88`.
- Normalize `GfxDriver/Drivers/Vulkan/VulkanPhysicalDevice.cpp:84-85`, where "is at least 1.2" is spelled `major > 1 || minor > 1`.

## PR 6 - Introduce a compiler backend seam, add Slang alongside glslang

Purely additive. [GfxDriver/ShaderCompiler/ShaderManager.h](GfxDriver/ShaderCompiler/ShaderManager.h) line 315 holds a concrete `GlslangWrapperUqPtr`; introduce a `ShaderCompilerBackend` interface (compile plus reflect) and add a Slang implementation selectable by flag, with glslang remaining the default. Map `GlslangIncluder` onto a Slang `ISlangFileSystem` over `Library`. Nothing changes for existing callers.

Scope depends on PR 0's second spike.

## PR 7 - Switch the default to Slang, delete ShaderRedecorator

Slang assigns descriptor sets and bindings across a composed program with multiple entry points and reports them through reflection, so the remaining redecoration, SPIRV-Cross patching via `get_binary_offset_for_decoration`, and `ResourceLimits` / `TBuiltInResource` all go away rather than being ported. This also resolves the never-completed TODO at `GlslangWrapper.cpp:198` about sourcing limits from the device, since limit validation moves to the driver.

Keep `ShaderReflection` unchanged and populate it from Slang reflection. Four behaviours need deliberate reimplementation:

- Clash diagnostics quality from `reserveBindings` / `allocateBindings`
- The struct-flattening naming convention at `ShaderRedecorator.cpp:306-355`: dotted names for nested UBO members, bare names for SSBO members, depended on verbatim by `Material` and the Vega property writers
- `validate_buffer_attr_type`, which whitelists only scalar int/uint/int64/uint64/float/double; decide whether to carry the restriction forward
- Push constants, currently hand-maintained in [GfxDriver/Pipeline/PushConstantRanges.cpp](GfxDriver/Pipeline/PushConstantRanges.cpp) and absent from reflection, come free here

## PR 8+ - Migrate shader sources to Slang modules and generics

Largest effort, biggest long-term payoff, safely deferrable. Slang interfaces plus generics with link-time specialization replace both the subroutine mechanism from PR 3 and the string-splicing `Builder` itself, which is the root source of the complexity this subsystem exists to manage. Migrate per-shader or per-family behind the PR 6 flag rather than as a flag day.

## Open risks

- Whether glslang's default GLSL I/O resolver assigns bindings coherently across stages in a single `TProgram` while respecting explicit bindings. Gates PR 4's primary approach; the fallback is documented above.
- How many of the ~130 shader sources clear Slang's GLSL subset. Gates PR 6 onward.

PRs 1 through 3 are worth doing regardless of how either spike resolves.
