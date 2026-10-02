---
name: Remaining Slang migration work
overview: A standalone working document for the pieces of the glslang-to-Slang migration that have not yet landed, with the dependencies between them made explicit. Six remain. Four items originally filed under PR 7 needed nothing from Slang and landed separately, as did one unscheduled cleanup they turned up. The completed work is summarised only briefly, except where it established something a later PR has to reproduce; the full archaeology stays in the existing plan.
todos:
  - id: osrb
    content: Slang OSRB approval and attribution. External gate on merging PR 6 onward; start now, independent of all code work
    status: pending
  - id: spike-slang-glsl
    content: "PR 0b: Compile the shader corpus through slangc -allow-glsl and classify failures, reporting Tests/ and non-Tests/ counts separately. Prefer assembled final_glsl over raw sources. No dependencies; gates the scope of PRs 6, 7 and 8"
    status: pending
  - id: remove-tshaderirutils
    content: "PR 3: Delete TShaderIRUtils and reimplement subroutine binding as a textual pass, resolving targets through the Library because the assembled source holds #include directives rather than their text. Landed as 9b8be38b08, no golden changed. Retired one copied header rather than the three estimated"
    status: completed
  - id: compile-material-at-once
    content: "PR 4a: Give glslToSpirv and buildSpirv a whole material at a time, with a program per stage and one shared I/O resolver. Landed as d2fbb038a8, no behaviour change and no golden moved. Established that one TProgram cannot span the stages, because a material may hold two shaders of the same stage"
    status: completed
  - id: remove-iomapresolver
    content: "PR 4b: Replace IoMapResolver with glslang's TDefaultGlslIoResolver, delete the three sentinels and ShaderRedecorator's allocation and SPIR-V patching, and re-aim the clash check at verifying glslang's output. Landed as f0bea7b8b7. Two of the nine goldens moved, and it corrected a long-standing bug that gave an array of resources one binding per element. Did not unblock PR 5c part 3 after all"
    status: completed
  - id: drop-copy-block
    content: "PR 5c part 3: Delete the 13-header copy block and the two mkdir guards from common-functions.sh. Now needs PR 7, not PR 4b: GlslangWrapper.cpp still includes iomapper.h for TGlslIoMapper and TDefaultGlslIoResolver, which are declared nowhere public. Verify with a full deps rebuild, since these 13 are genuinely not installed by glslang"
    status: pending
  - id: instance-1.4
    content: "PR 5d (optional): Raise kVulkanInstanceApiVersion to Vulkan 1.4. No dependencies and nothing depends on it. No technical upside; costs the NVIDIA driver floor rising from 535 to roughly R570. Take only if something else needs 1.4"
    status: pending
  - id: backend-seam
    content: "PR 6: Introduce a ShaderCompilerBackend interface behind ShaderManager.h:315's concrete GlslangWrapperUqPtr and add a flag-gated Slang implementation with ISlangFileSystem over Library. Needs PR 0b for scoping and OSRB to merge"
    status: pending
  - id: slang-default
    content: "PR 7: Switch the default to Slang, delete ShaderRedecorator and ResourceLimits, populate ShaderReflection from Slang reflection, and reimplement clash diagnostics, struct-flattening names, the type whitelist, array dimensions and push constants. Needs PR 6; smaller now that PR 4b has landed. Also required by PR 5c part 3"
    status: pending
  - id: shader-migration
    content: "PR 8+: Migrate shader sources to Slang modules and generics incrementally behind the PR 6 flag, retiring the string-splicing Builder and superseding PR 3's textual subroutine pass"
    status: pending
  - id: varying-check
    content: "Non-Slang: Reject an inter-stage varying with no explicit location. Landed as f335583933. Three exemptions were needed, each of which rejected real shaders before it was added: vertex inputs and fragment outputs, built-in members of gl_PerVertex, and blocks carrying one location for the whole block"
    status: completed
  - id: createcache-test
    content: "Non-Slang: Cover ShaderManager::createCache. Landed as cb950d202f, pinning the empty reflection as the documented contract rather than making it populate one"
    status: completed
  - id: reflect-array-dim
    content: "Non-Slang: Let ShaderReflection express an array dimension on a buffer member. Landed as b4c9acc131. Found a live two-dimensional member, PPLL's batch_stats, which is flattened to element count and element stride to match how PPLLRender reads it back"
    status: completed
  - id: reflect-push-constants
    content: "Non-Slang: Reflect push constants and check every push against them. Landed as 02527b6f9b. The dead name parameter on setPushConstants became the key. Deriving the pipeline-creation ranges from reflection is left to PR 7"
    status: completed
  - id: dead-accum-composite
    content: "Cleanup: delete the dead accumulationComposite.frag. Landed as 31cc64861c, two files and 25 deletions. Its last consumer went with distributed mode in f20941c8ef, leaving only the source and its ShaderManifest.json entry"
    status: completed
isProject: false
---

# Remaining work: replacing glslang with Slang

Companion to [.cursor/plans/glslang-to-slang-migration.md](.cursor/plans/glslang-to-slang-migration.md), which keeps the full history and the reasoning behind decisions already taken. This file covers only what is left.

## What has already landed

- **PR 0a** - Spiked and kept the SDK bump. The tree builds and runs on Vulkan SDK 1.4.363.0 with glslang 16.6.0. This disproved the plan's central premise: the private headers were never what pinned the SDK, so PRs 3 and 4 do not gate anything except their own cleanup.
- **PR 1** (`95fb438ec0`) - Replaced byte-exact SPIR-V goldens with `ShaderReflection` text goldens plus `spirv-val`. Nine `.reflect` files. This is the safety net every remaining PR is verified against.
- **PR 2** (`34eed87b9a`) - Hygiene: deleted `kShaderStageCount`, hardened glslang error-log scraping, collapsed the vestigial multi-set scaffolding, restored the rule of zero on `ShaderReflection`.
- **PR 5a** - Vulkan SDK 1.4.363.0 with Slang built alongside glslang. `slangc` and `slang.h` are now in the deps prefix.
- **PR 5b** (`4752a50653`) - Retargeted the compiler to Vulkan 1.3 / SPIR-V 1.6, split `VULKAN_API_VERSION` into instance and device-minimum constants, fixed a `VulkanPhysicalDevice` gate that tested 1.2 while chaining 1.3 structs.
- **PR 5c parts 1 and 2** (`d4019d84e6`) - Deleted the dead `build_info.h` include and swapped glslang's disassembler for the SPIRV-Tools one. The copy block in `scripts/common-functions.sh` is down from 16 files to 14.
- **PR 3** (`9b8be38b08`) - Deleted `TShaderIRUtils` and moved subroutine resolution to a textual pass in `ShaderManager`, run before glslang sees the source. Took the copy block from 14 files to 13.
- **PR 4a** (`d2fbb038a8`) - Restructured `glslToSpirv` and `buildSpirv` to take a whole material at a time, giving each stage its own `glslang::TProgram` but sharing one I/O resolver between them. Pure plumbing: the resolver is still our stateless `IoMapResolver`, so no behaviour changed and no golden moved.
- **PR 4b** (`f0bea7b8b7`) - Handed binding and location assignment to `glslang::TDefaultGlslIoResolver`, reducing `ShaderRedecorator` to reflection extraction plus a clash check and deleting the sentinels, the allocator and the SPIR-V patching. Two of the nine goldens moved. Corrected a long-standing bug in which an array of resources was given one binding per element rather than a single binding with a descriptor count.

### Four items taken outside the PR sequence

Each was filed under PR 7 but needs nothing from Slang, and each is better as its own commit than as a line in PR 7's diff: the golden movement stays attributable, and the behaviour is pinned by tests before Slang has to reproduce it. All four have landed.

- **Varying locations** (`f335583933`) - Rejects an inter-stage varying that arrives without an explicit location, which glslang would otherwise match across stages only by luck of declaration order. Three exemptions were needed and each one rejected real shaders before it was added: vertex inputs and fragment outputs, which are matched against the reflection rather than another stage; built-in members of the implicit `gl_PerVertex` block, which is what made the first attempt reject all five vertex shaders; and blocks carrying one location for the whole block, the form every mesh and task shader uses.
- **`createCache` coverage** (`cb950d202f`) - Pins the empty reflection as the documented contract rather than making the method populate one. Its only caller, `saveArtifacts`, reads just the GLSL and SPIR-V, so a regression would have shown up as missing artifacts rather than a failing test.
- **Array dimensions** (`b4c9acc131`) - `ItemInfo` gained `array_length` and `array_stride`, so element *i* of a buffer member is at `offset + i * array_stride`. See below for the two-dimensional member this turned up.
- **Push constants** (`02527b6f9b`) - Reflected into new `push_constants` and `push_constant_attrs` sections, and every push is now checked against them. See below.

### One cleanup they turned up

- **Deleted the dead `accumulationComposite.frag`** (`31cc64861c`) - Found while reflecting push constants, as a block nothing ever pushes. Its last consumer was `DistributedCompositor::initResources`, so removing distributed mode in `f20941c8ef` deleted the caller and dropped `distributedCompositor.frag` from the manifest but left this one compiling into the shader library with no material to instantiate it. `MultiGpuCompositor` had been its other consumer until `5612fa7547` removed the compositor-side accum texture array in 2021; the live sibling there is the peer variant. Two files, the source and its `ShaderManifest.json` entry, that file being the only thing that decides what enters the library.

  Worth carrying into PR 8's sweeps: the name is a prefix of `accumulationCompositePeer.frag`, so a plain search for it matches the live sibling and it reads as referenced when it is not. Anything that walks the library looking for dead shaders needs a word-boundary match, not a substring one.

## How the remaining pieces depend on each other

```mermaid
flowchart TD
    OSRB["Slang OSRB approval (external, start now)"]
    PR0b["PR 0b: slangc -allow-glsl spike"]
    PR5c3["PR 5c part 3: delete the 13-header copy block"]
    PR5d["PR 5d: instance to Vulkan 1.4 (optional)"]
    PR6["PR 6: ShaderCompilerBackend seam, Slang alongside"]
    PR7["PR 7: Slang becomes the default"]
    PR8["PR 8+: migrate shaders to Slang modules"]

    PR0b --> PR6
    OSRB --> PR6
    PR6 --> PR7
    PR6 --> PR8
    PR7 --> PR5c3
```

What is left is one track, plus one standalone item:

- **Migration track: PRs 0b, 6, 7, 8+.** The actual replacement. Gated at the front by the spike and by OSRB approval.
- **Cleanup tail: PR 5c part 3.** Now the last thing rather than an independent track, because only PR 7 can retire the private headers. See below.
- **Standalone: PR 5d.** Independent of everything, and optional.

The cleanup track no longer exists as a separate thing. It was meant to be PRs 3, 4a and 4b retiring the private headers between them; all three have landed and the headers are still there. PR 3 took one of the thirteen and PR 4b took none, because handing assignment to glslang needs more of `iomapper.h` than subclassing it did, not less.

## PR 0b - Spike slangc's GLSL compatibility mode

**Depends on:** nothing. Unblocked since PR 5a installed `slangc` at `$PREFIX/bin/slangc`.
**Gates:** the scope of PRs 6, 7 and 8.

Compile the shader corpus through `slangc -allow-glsl` and classify each file three ways: compiles clean, fails on something mechanical and pattern-fixable, or fails on something Slang's GLSL subset does not model (which forces that shader into the PR 8 bucket early).

Two things make the raw-source version of this spike misleading:

- Stage files are not independently compilable. They depend on the `#include` dictionary assembled at runtime by `buildExtensionAndIncludesString` and on the `processOperators` substitutions. Either run over assembled `final_glsl` captured from a real run via the `ShaderArtifactTypeBits` hooks, or treat a raw pass as a lower bound only.
- Report test and non-test counts separately. Of 163 shader sources, **88 live under `Tests/`**, so more than half the corpus is fixtures that can be rewritten or deleted freely. Only the 75 outside `Tests/` are load-bearing. Recount rather than trusting those figures: the test half grows as coverage is added, and the non-test half shrank by one when `accumulationComposite.frag` went.

## What PR 3 established, which the rest inherits

Two findings from landing it change how later PRs should be scoped.

**The assembled source does not contain its includes.** `buildExtensionAndIncludesString` emits `#include` directives, because `Library` wraps each dictionary entry as `"#include \"...\""`, and `GlslangIncluder` resolves them during parse. So `final_glsl` names its includes without holding their text. The subroutine pass has to look targets up through the Library, since the colour conversion subroutines are defined only in an included file. Any backend that replaces glslang inherits this: it needs its own answer for the include dictionary, not just for the assembled string.

**Textual inspection of the shader library has to survive the preprocessor.** `quantitativeScaleTemplate.vert` selects the type of a function's last parameter with an `#if`, and in one case leaves the body on the far side of the `#endif`. Neither a parameter-list pattern nor brace-matching can classify those definitions; what works is that a definition names its return type immediately before the function name. Worth remembering before writing any further pass over GLSL text.

## What PR 4a established, which PR 4b builds on

The original plan for PR 4 was to link every stage of a material into one `glslang::TProgram` and call the public `mapIO(nullptr, nullptr)`. Both halves of that were wrong, and reading glslang 16.6.0's `iomapper.cpp` and `ShaderLang.cpp` is what settled it.

**One `TProgram` cannot span a material's stages.** A material may hold two shaders of the same stage, and `RaytracingTest` does it twice: `InstancingTest` pairs `Instancing.miss` with the shared `Shading/rtShadowsSimple.miss`, and `BlasMultiMesh` does the same. A program has room for one intermediate per stage, so linking those together would merge both miss shaders and return the same module for both caches, if it linked at all with two `main` bodies in one stage. PR 4a therefore gives each stage its own program.

**A null resolver does not give you a program-wide one.** `TIoMapper::addStage` constructs its `TDefaultIoResolver` as a stage-local, so `mapIO(nullptr, nullptr)` allocates bindings independently per stage, which is the collision PR 4 exists to prevent. Pairing a null resolver with `TGlslIoMapper` is worse: `TProgram::mapIO` forwards the raw pointer to `doMap`, which dereferences it on its first line.

**Sharing the resolver is what actually delivers cross-stage coherence.** `TDefaultGlslIoResolver` keeps its slot maps in the resolver rather than the program, and none of `beginResolve`, `endResolve`, `beginCollect` or `endCollect` clears them; all four only track which stage is current. One instance reused across sequential per-stage `mapIO` calls accumulates bindings by name exactly as a single program would. PR 4a put that sharing in place with our existing stateless `IoMapResolver`, so PR 4b is only a swap.

**Lifetime constraint that swap must respect.** A stateful resolver remembers names in a `TString`, which belongs to the pool of whichever `TProgram` was current when it was recorded, and that pool dies with its program. `glslToSpirv` holds all the shaders and programs for the whole call and declares the resolver after the programs so that it is destroyed first. Do not reorder those declarations.

## What PR 4b established, which PR 7 inherits

Four findings, three of them constraints on anything that touches this code again.

**glslang is built without RTTI, so none of its classes can be subclassed.** The typeinfo a derived class's own typeinfo refers to was never emitted, and the link fails with `undefined reference to typeinfo for glslang::TDefaultGlslIoResolver` on every target. The tell is that *constructing* the class links fine: the vtable is there, only the typeinfo is missing. The pre-4b `IoMapResolver` got away with deriving from `TIoMapResolver` only because that class is pure-abstract with no out-of-line virtual to anchor its typeinfo, so the compiler emits it weakly on our side. `IoMapResolver` therefore holds a `TDefaultGlslIoResolver` and forwards nineteen methods to it. **Check this first if PR 6 finds itself wanting to subclass anything in Slang.**

**An array of resources is one binding, not one per element.** `ShaderRedecorator` had always claimed a binding per element, so arrays left gaps after them. `VulkanMaterial` has meanwhile always built a single `VkDescriptorSetLayoutBinding` with `descriptorCount` set to the array size, so the runtime only ever used the first binding of each array and the rest were silently wasted. glslang assigns the Vulkan way, which is what surfaced it: `numBindings` in `TDefaultGlslIoResolver::resolveBinding` is the cumulative array size only when the target is OpenGL. The array size still belongs in the reflection, which is where the descriptor count comes from; it just has nothing to do with occupancy.

**Inter-stage varyings must declare explicit locations, and blocks must declare them per member.** Both are already true of the whole shader library, and both are now load-bearing. A block whose members carry locations must not also get one on the block variable, which is why `IoMapResolver::resolveInOutLocation` declines to assign to structs; glslang only skips blocks whose first member is a built-in, and the Vulkan path in `TDefaultIoResolverBase` behaves the same way. Separately, glslang's own comment on that Vulkan path is that it "does not do proper cross-stage lining up", so a varying *without* an explicit location would be matched across stages only by luck of declaration order. Nothing checks for that; adding a check is cheap if it ever seems worth it.

**glslang assigns only to resources it considers live.** `resolveBinding` auto-assigns when `ent.live && doAutoBindingMapping()`, and the shader library routinely declares resources a given stage never reads. Since `ShaderReflection` is what `VulkanMaterial` binds descriptors from, an unassigned resource is both missing from the reflection and liable to collide, every unassigned one defaulting to binding zero. `IoMapResolver` forces the liveness flag, restoring the one behaviour the old resolver had that glslang's does not. This is the only liveness gate that matters; `resolveInOutLocation` has none.

Two things stayed true as planned. Clash detection could not be delegated: `reserveSlot` explicitly "tolerate[s] aliasing, by not double-recording aliases", and the only clash glslang reports is one name carrying different explicit bindings across stages, which is the opposite of what `ShaderRedecoratorClashTest` exercises. And `setShiftUboBinding` and friends remain a dead end, since `resolveBinding` adds the shift base to explicitly declared bindings too.

Only two of the nine goldens moved, rather than all of them as predicted, because most of the corpus binds explicitly. PR 4b also left `createCache` unverified, its one live caller `saveArtifacts` passing `nullptr` as the redecorator; `cb950d202f` has since covered it, pinning the empty reflection as the contract `ShaderManager.h` already documented.

## What the four non-Slang items established, which PR 7 inherits

All four are reflection behaviour, so PR 7 has to reproduce them from Slang rather than inherit them. The tests pin each one, so PR 7 will find out quickly, but the reimplementation surface is larger than PR 7's section below originally claimed.

**There is a live two-dimensional array member, and nothing had ever reflected it.** `PPLL_BATCH_STAT_COUNTERS_SSBO` declares `PPLLFragmentStats batch_stats[maxNumBatches][numTiles]`. The array-dimension work initially rejected multi-dimensional members with a `CHECK`, on the assumption that `slabs[64]` was the only array case; `QueryRendererTest` aborted on the first run. It is now flattened to total element count and element stride, which is how `PPLLRender` actually consumes it: the buffer is read back into one flat `std::vector<PPLLFragmentStats>` and indexed linearly. Flattening is only valid if the dimensions are tightly packed, so the exact division is checked. Two further cases are rejected because each would be a silently wrong number: a dimension given by a specialization constant, which SPIRV-Cross stores as that constant's id rather than its value, and a runtime-sized array with inner dimensions, which has no element count to divide by.

**SPIRV-Cross names a push constant block after its variable and a uniform block after its type.** This asymmetry cost a round trip. Reasoning from the uniform case is what produces the wrong answer: a `.reflect` file says `SLAB_ADDRESS_TABLE_UBO` rather than `slab_address_table`, so the type name looks like the universal rule, but `push_constant_buffers` entries carry the instance name instead. Both names are now recorded, since no call site should have to know which one a category reports, and either may be absent - a block declared without an instance name has no variable name, and an empty one must not be recorded or a lookup on `""` would succeed.

**A push constant block's range is its members' span, not its declared size.** The declared size always starts at zero, however the first member is offset. The fragment half of `PolyMark`'s split block begins at 4, and 4 is what the call site declares, so the span is the only definition that matches the hand-written ranges. Only the test shader covers this: both goldens that have push constants hold a single member at offset 0.

**`setPushConstants`'s `name` parameter was dead.** It was threaded through `CommandList`, the command struct and the executor, then dropped at `vkCmdPushConstants`. It is now the key into the reflection, and the pushed range must fall inside whatever it resolves to. Five call sites passed prose that could never resolve and now name the block or the member they cover. Stage bits are deliberately not checked, since the stage mask must match the pipeline layout and Vulkan enforces that itself.

## PR 5c part 3 - Delete the header copy block

**Depends on:** PR 7. This was expected to depend on PR 4b alone, and does not.

`GlslangWrapper.cpp` is the only consumer of all 13 copies. It includes just `iomapper.h` and `LiveTraverser.h` directly and reaches the other eleven through them, so whatever retires those two includes retires the lot at once.

PR 4b was supposed to be that, by removing the `TIoMapResolver` subclass. It had the opposite effect: `IoMapResolver` now needs `TGlslIoMapper` and `TDefaultGlslIoResolver` by name, and both are declared only in the private `iomapper.h`. The public `ShaderLang.h` has `TIoMapResolver`, and 16.6.0 did promote `TVarEntryInfo` into it, which is what made this look tractable, but it declares neither mapper nor any concrete resolver. The only concrete resolver in the private header is the OpenGL one; the Vulkan `TDefaultIoResolver` is defined inside `iomapper.cpp` and is unreachable by design, which is why `mapIO(nullptr, nullptr)` has to construct it internally.

So the headers survive until glslang itself goes, in PR 7. Delete the block and the two surviving `mkdir -p` guards at `scripts/common-functions.sh:1092-1108`, plus the comment at 1077-1082.

PR 3 was expected to take three of these and took one, `Include/InfoSink.h`. The rest survived because `LiveTraverser.h`, which PR 0a gave `GlslangWrapper.cpp` a direct include of, pulls in `Common.h`, `reflection.h`, `localintermediate.h` and `gl_types.h`, with `intermediate.h` and the rest of `Include/` behind those.

Unlike the two copies dropped in part 2, these 13 genuinely are not installed by glslang, so removing them really will take them out of the prefix. **Verify with a full deps rebuild**; the headers are already in the prefix, so an incremental build will mask the failure.

## PR 5d - Raise the instance to Vulkan 1.4 (optional)

**Depends on:** nothing. **Nothing depends on it.**

A one-line change to `kVulkanInstanceApiVersion`, plus `kMinVulkanDeviceApiVersion` if the device floor should move with it. Confirmed to have no technical upside:

- Vulkan 1.4 also tops out at SPIR-V 1.6, which PR 5b already targets.
- Nothing 1.4 promotes to core is used anywhere in the tree.
- It raises the NVIDIA driver floor from the enforced 535 (`VulkanPlatform.cpp:556`) to roughly R570.

Take it only if something else comes to need 1.4, and confirm the supported-driver matrix first.

## PR 6 - Introduce a compiler backend seam, add Slang alongside glslang

**Depends on:** PR 0b for scoping, and Slang OSRB approval to merge (not to develop).
**Required by:** PRs 7 and 8+.

Purely additive. `ShaderManager.h:315` holds a concrete `GlslangWrapperUqPtr glslang_wrapper_`. Introduce a `ShaderCompilerBackend` interface covering compile and reflect, add a Slang implementation selectable by flag, keep glslang the default. Map `GlslangIncluder` onto a Slang `ISlangFileSystem` over `Library`. Nothing changes for existing callers.

That file system is not optional detail: per the PR 3 findings above, the assembled source carries `#include` directives rather than their text, so a Slang backend without it sees shaders whose functions are largely undefined.

## PR 7 - Switch the default to Slang, delete ShaderRedecorator

**Depends on:** PR 6. Smaller now that PR 4b has reduced `ShaderRedecorator` to reflection extraction plus a clash check.
**Required by:** PR 5c part 3, which only this can unblock.

Slang assigns descriptor sets and bindings across a composed program with multiple entry points and reports them through reflection, so the remaining reflection extraction and `ResourceLimits` / `TBuiltInResource` go away rather than being ported. PR 4b already removed the SPIRV-Cross patching via `get_binary_offset_for_decoration`. This also closes the never-completed TODO in `GlslangWrapper.cpp` about sourcing limits from the device, since limit validation moves to the driver.

Keep `ShaderReflection` unchanged and populate it from Slang reflection. Seven behaviours need deliberate reimplementation, four of them newly pinned by the non-Slang commits rather than merely desirable:

- Clash diagnostics quality from `recordBinding`, which is all that is left of the old allocator and still carries the message `ShaderRedecoratorClashTest` pins.
- The struct-flattening naming convention: the block in `reflect_buffers` and the rule stated with it. Dotted names for nested UBO members, bare names for SSBO members, depended on verbatim by `Material` and the Vega property writers.
- `validate_buffer_attr_type`, which whitelists only scalar int/uint/int64/uint64/float/double. Decide whether to carry the restriction forward.
- **Array dimensions**, now `array_length` and `array_stride` on `ItemInfo`. The flattening rule for multi-dimensional members and the three cases it rejects are described above and asserted by `ShaderRedecoratorTest`.
- **Push constant reflection**, now `push_constants` and `push_constant_attrs`. The span rule, the dual naming and the anonymous-block case are all asserted.
- **The push range check** in `Material::validatePushConstantRange`, which every `setPushConstants` call now goes through. Whatever replaces the reflection has to keep resolving a name to a block or a member, or every push in the tree throws.
- **Explicit varying locations**, rejected in `GlslangWrapper.cpp` rather than in the redecorator, so this one moves with the backend rather than with `ShaderRedecorator`. Its three exemptions are the part to preserve; each was arrived at by a shader failing.

Two things that were on this list are no longer: the array-dimension gap and push constants both landed separately, so their golden movement is already absorbed rather than arriving inside PR 7's diff.

A better end state exists for push constants than what landed, and PR 7 is where it belongs: derive the ranges from reflection at pipeline creation and delete the nine hand-written ones. They reach a pipeline three different ways, by `setPushConstantRanges` on a descriptor, as an argument to `createComputePipeline`, and as one to the graphics and raytracing creators, so search for `PushConstantRange` rather than for any one of those. That is the version where the shader and the range cannot disagree at all, rather than one where disagreement is caught at the push. It was not done now because it is a change to pipeline creation, not to reflection.

Hard constraint carried over from PR 1: the shader library declares its uniform blocks `layout(std430)`, legal only because `uniformBufferStandardLayout` and `scalarBlockLayout` are hard device requirements. Slang needs the equivalent of `-fvk-use-scalar-layout` or per-block layout attributes rather than its defaults, or `spirv-val` will reject the output.

## PR 8+ - Migrate shader sources to Slang modules and generics

**Depends on:** PR 6 for the flag. Supersedes PR 3's textual subroutine pass.

Largest effort, biggest payoff, safely deferrable. Slang interfaces plus generics with link-time specialization replace both the subroutine mechanism and the string-splicing `Builder` that is the root source of this subsystem's complexity. Migrate per-shader or per-family behind the PR 6 flag rather than as a flag day.

## Risks that span the remaining work

- **Slang OSRB approval.** Not technical, entirely outside this plan's control, and a hard gate on shipping anything from PR 6 onward. Start it now rather than when the code is ready.
- **How much of the corpus clears Slang's GLSL subset.** Gates PR 6 onward; PR 0b measures it.
- **Whether glslang's resolver assigns coherently across stages.** Closed by PR 4b. A shared `TDefaultGlslIoResolver` keys its slot maps by name and nothing clears them between stages, and the render suite passes on its numbering. Vertex attribute locations do still start at 0.
- **Sentinel fragility.** Closed by PR 4b, which deleted them. They were derived from glslang's internal `layout*End` markers and survived 1.3.275 to 16.6.0 unchanged, so the exposure was latent rather than active, but it is now gone.
- **Private-header reach.** PRs 3, 4a and 4b were each expected to reduce our dependence on glslang's private headers and collectively retired one of thirteen. Treat any future estimate of this kind sceptically: the headers come as a transitive closure behind `LiveTraverser.h` and `iomapper.h`, so nothing short of dropping both includes changes the count at all.
- **PR 7's reimplementation surface keeps growing.** It started at five behaviours and is now seven, the four non-Slang commits having pinned as tests things that were previously undefined or simply absent. That is the right direction, since each is now specified rather than accidental, but PR 7 is not getting smaller in the way PR 4b's reduction of `ShaderRedecorator` suggested. Read the section above before scoping it.
- **Reasoning about SPIRV-Cross and glslang from one case.** Three errors this session came from generalising a single observation: that `slabs[64]` was the only array member, that resource names are always type names, and that nine goldens were ten. Each was caught by a build or a test within one cycle, so the exposure is cost rather than risk, but the pattern is worth naming - these libraries are not uniform across resource categories, and the cheap move is to check the category in question rather than infer it from a neighbour.