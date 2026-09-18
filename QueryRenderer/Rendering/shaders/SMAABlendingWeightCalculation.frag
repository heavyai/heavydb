/**
 * Copyright (C) 2013 Jorge Jimenez (jorge@iryoku.com)
 * Copyright (C) 2013 Jose I. Echevarria (joseignacioechevarria@gmail.com)
 * Copyright (C) 2013 Belen Masia (bmasia@unizar.es)
 * Copyright (C) 2013 Fernando Navarro (fernandn@microsoft.com)
 * Copyright (C) 2013 Diego Gutierrez (diegog@unizar.es)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to
 * do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software. As clarification, there
 * is no requirement that the copyright notice and permission be included in
 * binary distributions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * Blend Weight Calculation Vertex Shader
 */

#define saturate(a) clamp(a, 0.0, 1.0)

#if defined(SMAA_PRESET_LOW)
#define SMAA_MAX_SEARCH_STEPS 4
#define SMAA_DISABLE_DIAG_DETECTION
#elif defined(SMAA_PRESET_MEDIUM)
#define SMAA_MAX_SEARCH_STEPS 8
#define SMAA_DISABLE_DIAG_DETECTION
#elif defined(SMAA_PRESET_HIGH)
#define SMAA_MAX_SEARCH_STEPS 16
#define SMAA_MAX_SEARCH_STEPS_DIAG 8
#define SMAA_CORNER_ROUNDING 25
#elif defined(SMAA_PRESET_ULTRA)
#define SMAA_MAX_SEARCH_STEPS 32
#define SMAA_MAX_SEARCH_STEPS_DIAG 16
#define SMAA_CORNER_ROUNDING 25
#endif

/**
 * SMAA_CORNER_ROUNDING specifies how much sharp corners will be rounded.
 *
 * Range: [0, 100]
 *
 * Define SMAA_DISABLE_CORNER_DETECTION to disable corner processing.
 */
#ifndef SMAA_CORNER_ROUNDING
#define SMAA_CORNER_ROUNDING 25
#endif

//-----------------------------------------------------------------------------
// Non-Configurable Defines

#define SMAA_AREATEX_MAX_DISTANCE 16
#define SMAA_AREATEX_MAX_DISTANCE_DIAG 20
#define SMAA_AREATEX_PIXEL_SIZE (1.0 / vec2(160.0, 560.0))
#define SMAA_AREATEX_SUBTEX_SIZE (1.0 / 7.0)
#define SMAA_SEARCHTEX_SIZE vec2(66.0, 33.0)
#define SMAA_SEARCHTEX_PACKED_SIZE vec2(64.0, 16.0)
#define SMAA_CORNER_ROUNDING_NORM (float(SMAA_CORNER_ROUNDING) / 100.0)

//-----------------------------------------------------------------------------
// Texture Access Defines


//-----------------------------------------------------------------------------
// Global Uniforms

layout(std430) uniform SMAA_BLENDING_WEIGHT_CALCULATION_FRAG_UBO_TYPE {
  vec4 SMAA_RT_METRICS;
  vec4 FULL_SMAA_RT_METRICS;
};

layout(push_constant) uniform SMAA_EDGE_DETECT_PUSH_CONSTANTS {
  uint32_t subsampleIndicesIndex;
} pushConstants;

#define numSamples <numSamples>

#if numSamples == 1
const vec4 subsampleIndices[1] = {vec4(0,0,0,0)};
#elif numSamples == 2
const vec4 subsampleIndices[2] = {vec4(1,1,1,0), vec4(2,2,2,0)};
#elif numSamples == 4
const vec4 subsampleIndices[4] = {vec4(1,1,1,0), vec4(2,1,1,0), vec4(1,2,2,0), vec4(2,2,2,0)};
#endif

/**
 * Conditional move:
 */
void SMAAMovc(bvec2 cond, inout vec2 variable, vec2 value) {
    if (cond.x) variable.x = value.x;
    if (cond.y) variable.y = value.y;
}

//-----------------------------------------------------------------------------
// Diagonal Search Functions

#if !defined(SMAA_DISABLE_DIAG_DETECTION)

/**
 * Allows to decode two binary values from a bilinear-filtered access.
 */
vec2 SMAADecodeDiagBilinearAccess(vec2 e) {
    // Bilinear access for fetching 'e' have a 0.25 offset, and we are
    // interested in the R and G edges:
    //
    // +---G---+-------+
    // |   x o R   x   |
    // +-------+-------+
    //
    // Then, if one of these edge is enabled:
    //   Red:   (0.75 * X + 0.25 * 1) => 0.25 or 1.0
    //   Green: (0.75 * 1 + 0.25 * X) => 0.75 or 1.0
    //
    // This function will unpack the values (mad + mul + round):
    // wolframalpha.com: round(x * abs(5 * x - 5 * 0.75)) plot 0 to 1
    e.r = e.r * abs(5.0 * e.r - 5.0 * 0.75);
    return round(e);
}

vec4 SMAADecodeDiagBilinearAccess(vec4 e) {
    e.rb = e.rb * abs(5.0 * e.rb - 5.0 * 0.75);
    return round(e);
}

/**
 * These functions allows to perform diagonal pattern searches.
 */
vec2 SMAASearchDiag1(sampler2D edgesTex, vec2 texcoord, vec2 dir, out vec2 e) {
    vec4 coord = vec4(texcoord, -1.0, 1.0);
    vec3 t = vec3(FULL_SMAA_RT_METRICS.xy, 1.0);
    while (coord.z < float(SMAA_MAX_SEARCH_STEPS_DIAG - 1) &&
           coord.w > 0.9) {
        coord.xyz = fma(t, vec3(dir, 1.0), coord.xyz);
        e = textureLod(edgesTex, coord.xy, 0.0).rg;
        coord.w = dot(e, vec2(0.5, 0.5));
    }
    return coord.zw;
}

vec2 SMAASearchDiag2(sampler2D edgesTex, vec2 texcoord, vec2 dir, out vec2 e) {
    vec4 coord = vec4(texcoord, -1.0, 1.0);
    coord.x += 0.25 * FULL_SMAA_RT_METRICS.x; // See @SearchDiag2Optimization
    vec3 t = vec3(FULL_SMAA_RT_METRICS.xy, 1.0);
    while (coord.z < float(SMAA_MAX_SEARCH_STEPS_DIAG - 1) &&
           coord.w > 0.9) {
        coord.xyz = fma(t, vec3(dir, 1.0), coord.xyz);

        // @SearchDiag2Optimization
        // Fetch both edges at once using bilinear filtering:
        e = textureLod(edgesTex, coord.xy, 0.0).rg;
        e = SMAADecodeDiagBilinearAccess(e);

        // Non-optimized version:
        // e.g = textureLod(edgesTex, coord.xy, 0.0).g;
        // e.r = textureLodOffset(edgesTex, coord.xy, 0.0, ivec2(1, 0)).r;

        coord.w = dot(e, vec2(0.5, 0.5));
    }
    return coord.zw;
}

/**
 * Similar to SMAAArea, this calculates the area corresponding to a certain
 * diagonal distance and crossing edges 'e'.
 */
vec2 SMAAAreaDiag(sampler2D areaTex, vec2 dist, vec2 e, float offset) {
    vec2 texcoord = fma(vec2(SMAA_AREATEX_MAX_DISTANCE_DIAG, SMAA_AREATEX_MAX_DISTANCE_DIAG), e, dist);

    // We do a scale and bias for mapping to texel space:
    texcoord = fma(SMAA_AREATEX_PIXEL_SIZE, texcoord, 0.5 * SMAA_AREATEX_PIXEL_SIZE);

    // Diagonal areas are on the second half of the texture:
    texcoord.x += 0.5;

    // Move to proper place, according to the subpixel offset:
    texcoord.y += SMAA_AREATEX_SUBTEX_SIZE * offset;

    // Do it!
    return textureLod(areaTex, texcoord, 0.0).rg;
}

/**
 * This searches for diagonal patterns and returns the corresponding weights.
 */
vec2 SMAACalculateDiagWeights(sampler2D edgesTex, sampler2D areaTex, vec2 texcoord, vec2 e, vec4 subsampleIndices) {
    vec2 weights = vec2(0.0, 0.0);

    // Search for the line ends:
    vec4 d;
    vec2 end;
    if (e.r > 0.0) {
        d.xz = SMAASearchDiag1(edgesTex, texcoord, vec2(-1.0,  -1.0), end);
        d.x += float(end.y > 0.9);
    } else
        d.xz = vec2(0.0, 0.0);
    d.yw = SMAASearchDiag1(edgesTex, texcoord, vec2(1.0, 1.0), end);

    if (d.x + d.y > 2.0) { // d.x + d.y + 1 > 3
        // Fetch the crossing edges:
        vec4 coords = fma(vec4(-d.x + 0.25, -d.x, d.y, d.y + 0.25), FULL_SMAA_RT_METRICS.xyxy, texcoord.xyxy);
        vec4 c;
        c.xy = textureLodOffset(edgesTex, coords.xy, 0.0, ivec2(-1,  0)).rg;
        // c.xy = textureLodOffset(edgesTex, coords.xy, 0.0, ivec2(-1, -1)).rg; // croot fix
        c.zw = textureLodOffset(edgesTex, coords.zw, 0.0, ivec2( 1,  0)).rg;
        c.yxwz = SMAADecodeDiagBilinearAccess(c.xyzw);

        // Non-optimized version:
        // vec4 coords = fma(vec4(-d.x, -d.x, d.y, d.y), FULL_SMAA_RT_METRICS.xyxy, texcoord.xyxy);
        // vec4 c;
        // c.x = textureLodOffset(edgesTex, coords.xy, 0.0, ivec2(-1, 0)).g;
        // // c.x = textureLodOffset(edgesTex, coords.xy, 0.0, ivec2(-1,-1)).g; // croot fix
        // c.y = textureLodOffset(edgesTex, coords.xy, 0.0, ivec2( 0, 0)).r;
        // // c.y = textureLodOffset(edgesTex, coords.xy, 0.0, ivec2( 0,-1)).r; // croot fix
        // c.z = textureLodOffset(edgesTex, coords.zw, 0.0, ivec2( 1, 0)).g;
        // c.w = textureLodOffset(edgesTex, coords.zw, 0.0, ivec2( 1, 1)).r;

        // Merge crossing edges at each side into a single value:
        vec2 cc = fma(vec2(2.0, 2.0), c.xz, c.yw);

        // Remove the crossing edge if we didn't find the end of the line:
        SMAAMovc(bvec2(step(0.9, d.zw)), cc, vec2(0.0, 0.0));

        // Fetch the areas for this line:
        weights += SMAAAreaDiag(areaTex, d.xy, cc, subsampleIndices.z);
    }

    // Search for the line ends:
    d.xz = SMAASearchDiag2(edgesTex, texcoord, vec2(-1.0, 1.0), end);
    if (textureLodOffset(edgesTex, texcoord, 0.0, ivec2(1, 0)).r > 0.0) {
        d.yw = SMAASearchDiag2(edgesTex, texcoord, vec2(1.0, -1.0), end);
        d.y += float(end.y > 0.9);
    } else
        d.yw = vec2(0.0, 0.0);

    if (d.x + d.y > 2.0) { // d.x + d.y + 1 > 3
        // Fetch the crossing edges:
        // vec4 coords = fma(vec4(-d.x, -d.x, d.y, d.y), FULL_SMAA_RT_METRICS.xyxy, texcoord.xyxy);
        vec4 coords = fma(vec4(-d.x, d.x, d.y, -d.y), FULL_SMAA_RT_METRICS.xyxy, texcoord.xyxy);
        vec4 c;
        c.x  = textureLodOffset(edgesTex, coords.xy, 0.0, ivec2(-1,  0)).g;
        c.y  = textureLodOffset(edgesTex, coords.xy, 0.0, ivec2( 0,  1)).r;
        c.zw = textureLodOffset(edgesTex, coords.zw, 0.0, ivec2( 1,  0)).gr;
        // c.zw = textureLodOffset(edgesTex, coords.zw, 0.0, ivec2( 1, -1)).gr; // croot fix
        vec2 cc = fma(vec2(2.0, 2.0), c.xz, c.yw);

        // Remove the crossing edge if we didn't found the end of the line:
        SMAAMovc(bvec2(step(0.9, d.zw)), cc, vec2(0.0, 0.0));

        // Fetch the areas for this line:
        weights += SMAAAreaDiag(areaTex, d.xy, cc, subsampleIndices.w).gr;
    }

    return weights;
}
#endif


/**
 * Horizontal/vertical search functions for the 2nd pass.
 */

/**
 * This allows to determine how much length should we add in the last step
 * of the searches. It takes the bilinearly interpolated edge (see
 * @PSEUDO_GATHER4), and adds 0, 1 or 2, depending on which edges and
 * crossing edges are active.
 */
float SMAASearchLength(sampler2D searchTex, vec2 e, float offset) {
    // The texture is flipped vertically, with left and right cases taking half
    // of the space horizontally:
    vec2 scale = SMAA_SEARCHTEX_SIZE * vec2(0.5, -1.0);
    vec2 bias = SMAA_SEARCHTEX_SIZE * vec2(offset, 1.0);

    // Scale and bias to access texel centers:
    scale += vec2(-1.0,  1.0);
    bias  += vec2( 0.5, -0.5);

    // Convert from pixel coordinates to texcoords:
    // (We use SMAA_SEARCHTEX_PACKED_SIZE because the texture is cropped)
    scale *= 1.0 / SMAA_SEARCHTEX_PACKED_SIZE;
    bias *= 1.0 / SMAA_SEARCHTEX_PACKED_SIZE;

    // Lookup the search texture:
    return textureLod(searchTex, fma(scale, e, bias), 0.0).r;
}


float SMAASearchXLeft(sampler2D edgesTex, sampler2D searchTex, vec2 texcoord, float end) {
    /**
     * @PSEUDO_GATHER4
     * This texcoord has been offset by (-0.25, 0.125) in the vertex shader to
     * sample between edge, thus fetching four edges in a row.
     * Sampling with different offsets in each direction allows to disambiguate
     * which edges are active from the four fetched ones.
     */
    vec2 e = vec2(0.0, 1.0);
    while (texcoord.x > end &&
           e.g > 0.8281 && // Is there some edge not activated?
           e.r == 0.0) { // Or is there a crossing edge that breaks the line?
        e = textureLod(edgesTex, texcoord, 0.0).rg;
        texcoord = fma(-vec2(2.0, 0.0), FULL_SMAA_RT_METRICS.xy, texcoord);
    }

    float offset = fma(-(255.0 / 127.0), SMAASearchLength(searchTex, e, 0.0), 3.25);
    return fma(FULL_SMAA_RT_METRICS.x, offset, texcoord.x);

    // Non-optimized version:
    // We correct the previous (-0.25, -0.125) offset we applied:
    // texcoord.x += 0.25 * FULL_SMAA_RT_METRICS.x;

    // The searches are bias by 1, so adjust the coords accordingly:
    // texcoord.x += FULL_SMAA_RT_METRICS.x;

    // Disambiguate the length added by the last step:
    // texcoord.x += 2.0 * FULL_SMAA_RT_METRICS.x; // Undo last step
    // texcoord.x -= FULL_SMAA_RT_METRICS.x * (255.0 / 127.0) * SMAASearchLength(searchTex, e, 0.0);
    // return fma(FULL_SMAA_RT_METRICS.x, offset, texcoord.x);
}

float SMAASearchXRight(sampler2D edgesTex, sampler2D searchTex, vec2 texcoord, float end) {
    vec2 e = vec2(0.0, 1.0);
    while (texcoord.x < end &&
           e.g > 0.8281 && // Is there some edge not activated?
           e.r == 0.0) { // Or is there a crossing edge that breaks the line?
        e = textureLod(edgesTex, texcoord, 0.0).rg;
        texcoord = fma(vec2(2.0, 0.0), FULL_SMAA_RT_METRICS.xy, texcoord);
    }
    float offset = fma(-(255.0 / 127.0), SMAASearchLength(searchTex, e, 0.5), 3.25);
    return fma(-FULL_SMAA_RT_METRICS.x, offset, texcoord.x);
}

float SMAASearchYUp(sampler2D edgesTex, sampler2D searchTex, vec2 texcoord, float end) {
    vec2 e = vec2(1.0, 0.0);
    while (texcoord.y < end &&
           e.r > 0.8281 && // Is there some edge not activated?
           e.g == 0.0) { // Or is there a crossing edge that breaks the line?
        e = textureLod(edgesTex, texcoord, 0.0).rg;
        texcoord = fma(vec2(0.0, 2.0), FULL_SMAA_RT_METRICS.xy, texcoord);
    }
    float offset = fma(-(255.0 / 127.0), SMAASearchLength(searchTex, e.gr, 0.0), 3.25);
    return fma(-FULL_SMAA_RT_METRICS.y, offset, texcoord.y);
}

float SMAASearchYDown(sampler2D edgesTex, sampler2D searchTex, vec2 texcoord, float end) {
    vec2 e = vec2(1.0, 0.0);
    while (texcoord.y > end &&
           e.r > 0.8281 && // Is there some edge not activated?
           e.g == 0.0) { // Or is there a crossing edge that breaks the line?
        e = textureLod(edgesTex, texcoord, 0.0).rg;
        texcoord = fma(-vec2(0.0, 2.0), FULL_SMAA_RT_METRICS.xy, texcoord);
    }
    float offset = fma(-(255.0 / 127.0), SMAASearchLength(searchTex, e.gr, 0.5), 3.25);
    return fma(FULL_SMAA_RT_METRICS.y, offset, texcoord.y);
}

/**
 * Ok, we have the distance and both crossing edges. So, what are the areas
 * at each side of current edge?
 */
vec2 SMAAArea(sampler2D areaTex, vec2 dist, float e1, float e2, float offset) {
    // Rounding prevents precision errors of bilinear filtering:
    vec2 texcoord = fma(vec2(SMAA_AREATEX_MAX_DISTANCE, SMAA_AREATEX_MAX_DISTANCE), round(4.0 * vec2(e1, e2)), dist);

    // We do a scale and bias for mapping to texel space:
    texcoord = fma(SMAA_AREATEX_PIXEL_SIZE, texcoord, 0.5 * SMAA_AREATEX_PIXEL_SIZE);

    // Move to proper place, according to the subpixel offset:
    texcoord.y = fma(SMAA_AREATEX_SUBTEX_SIZE, offset, texcoord.y);

    // Do it!
    return textureLod(areaTex, texcoord, 0.0).rg;
}

//-----------------------------------------------------------------------------
// Corner Detection Functions

void SMAADetectHorizontalCornerPattern(sampler2D edgesTex, inout vec2 weights, vec4 texcoord, vec2 d) {
    #if !defined(SMAA_DISABLE_CORNER_DETECTION)
    vec2 leftRight = step(d.xy, d.yx);
    vec2 rounding = (1.0 - SMAA_CORNER_ROUNDING_NORM) * leftRight;

    rounding /= leftRight.x + leftRight.y; // Reduce blending for pixels in the center of a line.

    vec2 factor = vec2(1.0, 1.0);
    factor.x -= rounding.x * textureLodOffset(edgesTex, texcoord.xy, 0.0, ivec2(0,  -1)).r;
    factor.x -= rounding.y * textureLodOffset(edgesTex, texcoord.zw, 0.0, ivec2(1,  -1)).r;
    factor.y -= rounding.x * textureLodOffset(edgesTex, texcoord.xy, 0.0, ivec2(0, 2)).r;
    factor.y -= rounding.y * textureLodOffset(edgesTex, texcoord.zw, 0.0, ivec2(1, 2)).r;

    weights *= saturate(factor);
    #endif
}

void SMAADetectVerticalCornerPattern(sampler2D edgesTex, inout vec2 weights, vec4 texcoord, vec2 d) {
    #if !defined(SMAA_DISABLE_CORNER_DETECTION)
    vec2 leftRight = step(d.xy, d.yx);
    vec2 rounding = (1.0 - SMAA_CORNER_ROUNDING_NORM) * leftRight;

    rounding /= leftRight.x + leftRight.y;

    vec2 factor = vec2(1.0, 1.0);
    factor.x -= rounding.x * textureLodOffset(edgesTex, texcoord.xy, 0.0, ivec2( 1, 0)).g;
    factor.x -= rounding.y * textureLodOffset(edgesTex, texcoord.zw, 0.0, ivec2( 1, -1)).g;
    factor.y -= rounding.x * textureLodOffset(edgesTex, texcoord.xy, 0.0, ivec2(-2, 0)).g;
    factor.y -= rounding.y * textureLodOffset(edgesTex, texcoord.zw, 0.0, ivec2(-2, -1)).g;

    weights *= saturate(factor);
    #endif
}

//-----------------------------------------------------------------------------
// Blending Weight Calculation Pixel Shader (Second Pass)

vec4 SMAABlendingWeightCalculation(vec2 texcoord,
                                   vec2 pixcoord,
                                   vec4 offset[3],
                                   sampler2D edgesTex,
                                   sampler2D areaTex,
                                   sampler2D searchTex,
                                   vec4 subsampleIndices) { // Just pass zero for SMAA 1x, see @SUBSAMPLE_INDICES.
    vec4 weights = vec4(0.0, 0.0, 0.0, 0.0);

    vec2 e = texture(edgesTex, texcoord).rg;

    if (e.g > 0.0) { // Edge at north
        #if !defined(SMAA_DISABLE_DIAG_DETECTION)
        // Diagonals have both north and west edges, so searching for them in
        // one of the boundaries is enough.
        weights.rg = SMAACalculateDiagWeights(edgesTex, areaTex, texcoord, e, subsampleIndices);

        // We give priority to diagonals, so if we find a diagonal we skip
        // horizontal/vertical processing.
        if (weights.r == -weights.g) { // weights.r + weights.g == 0.0
        #endif

        vec2 d;

        // Find the distance to the left:
        vec3 coords;
        coords.x = SMAASearchXLeft(edgesTex, searchTex, offset[0].xy, offset[2].x);
        coords.y = offset[1].y; // offset[1].y = texcoord.y - 0.25 * FULL_SMAA_RT_METRICS.y (@CROSSING_OFFSET)
        d.x = coords.x;

        // Now fetch the left crossing edges, two at a time using bilinear
        // filtering. Sampling at -0.25 (see @CROSSING_OFFSET) enables to
        // discern what value each edge has:
        float e1 = textureLod(edgesTex, coords.xy, 0.0).r;

        // Find the distance to the right:
        coords.z = SMAASearchXRight(edgesTex, searchTex, offset[0].zw, offset[2].y);
        d.y = coords.z;

        // We want the distances to be in pixel units (doing this here allow to
        // better interleave arithmetic and memory accesses):
        d = abs(round(fma(FULL_SMAA_RT_METRICS.zz, d, -pixcoord.xx)));

        // SMAAArea below needs a sqrt, as the areas texture is compressed
        // quadratically:
        vec2 sqrt_d = sqrt(d);

        // Fetch the right crossing edges:
        float e2 = textureLodOffset(edgesTex, coords.zy, 0.0, ivec2(1, 0)).r;

        // Ok, we know how this pattern looks like, now it is time for getting
        // the actual area:
        weights.rg = SMAAArea(areaTex, sqrt_d, e1, e2, subsampleIndices.y);

        // Fix corners:
        coords.y = texcoord.y;
        SMAADetectHorizontalCornerPattern(edgesTex, weights.rg, coords.xyzy, d);

        #if !defined(SMAA_DISABLE_DIAG_DETECTION)
        } else {
            e.r = 0.0; // Skip vertical processing.
        }
        #endif
    }

    if (e.r > 0.0) { // Edge at west
        vec2 d;

        // Find the distance to the top:
        vec3 coords;
        coords.y = SMAASearchYUp(edgesTex, searchTex, offset[1].xy, offset[2].z);
        coords.x = offset[0].x; // offset[1].x = texcoord.x - 0.25 * FULL_SMAA_RT_METRICS.x;
        d.x = coords.y;

        // Fetch the top crossing edges:
        float e1 = textureLod(edgesTex, coords.xy, 0.0).g;

        // Find the distance to the bottom:
        coords.z = SMAASearchYDown(edgesTex, searchTex, offset[1].zw, offset[2].w);
        d.y = coords.z;

        // We want the distances to be in pixel units:
        d = abs(round(fma(FULL_SMAA_RT_METRICS.ww, d, -pixcoord.yy)));

        // SMAAArea below needs a sqrt, as the areas texture is compressed
        // quadratically:
        vec2 sqrt_d = sqrt(d);

        // Fetch the bottom crossing edges:
        float e2 = textureLodOffset(edgesTex, coords.xz, 0.0, ivec2(0, -1)).g;

        // Get the area for this direction:
        weights.ba = SMAAArea(areaTex, sqrt_d, e1, e2, subsampleIndices.x);

        // Fix corners:
        coords.x = texcoord.x;
        SMAADetectVerticalCornerPattern(edgesTex, weights.ba, coords.xyxz, d);
    }

    return weights;
}




void SMAABlendingWeightComputeOffsets(vec2 texcoord,
                                   out vec2 pixcoord,
                                   out vec4 offset[3]) {
    pixcoord = texcoord * FULL_SMAA_RT_METRICS.zw;

    // We will use these offsets for the searches later on (see @PSEUDO_GATHER4):
    offset[0] = fma(FULL_SMAA_RT_METRICS.xyxy, vec4(-0.25, 0.125,  1.25, 0.125), texcoord.xyxy);
    offset[1] = fma(FULL_SMAA_RT_METRICS.xyxy, vec4(-0.125, 0.25, -0.125,  -1.25), texcoord.xyxy);

    // And these for the searches, they indicate the ends of the loops:
    offset[2] = fma(FULL_SMAA_RT_METRICS.xxyy,
                    vec4(-2.0, 2.0, 2.0, -2.0) * float(SMAA_MAX_SEARCH_STEPS),
                    vec4(offset[0].xz, offset[1].yw));
}

layout(location = 0) out vec4 color;

uniform sampler2D edgeTex;
uniform sampler2D areaTex;
uniform sampler2D searchTex;

layout(location = 0) in vec2 fTexCoord;

void main(void)
{
    vec2 pixcoord;
    vec4 offsets[3];
    offsets[0] = vec4(0.0, 0.0, 0.0, 0.0);
    offsets[1] = vec4(0.0, 0.0, 0.0, 0.0);
    offsets[2] = vec4(0.0, 0.0, 0.0, 0.0);
    SMAABlendingWeightComputeOffsets(fTexCoord, pixcoord, offsets);

    color = SMAABlendingWeightCalculation(fTexCoord,
                                   pixcoord, // TODO(croot): can we just use gl_FragCoord?
                                   offsets,
                                   edgeTex,
                                   areaTex,
                                   searchTex,
                                   subsampleIndices[pushConstants.subsampleIndicesIndex]); // Just pass zero for SMAA 1x, see @SUBSAMPLE_INDICES.
}
