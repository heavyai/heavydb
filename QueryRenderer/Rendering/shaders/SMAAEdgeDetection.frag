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
 * Edge Detection Fragment Shader
 */

//-----------------------------------------------------------------------------
// SMAA Presets

/**
 * Note that if you use one of these presets, the following configuration
 * macros will be ignored if set in the " Configurable Defines " section.
 */

#if defined(SMAA_PRESET_LOW)
#define SMAA_THRESHOLD 0.15
#elif defined(SMAA_PRESET_MEDIUM)
#define SMAA_THRESHOLD 0.1
#elif defined(SMAA_PRESET_HIGH)
#define SMAA_THRESHOLD 0.1
#elif defined(SMAA_PRESET_ULTRA)
#define SMAA_THRESHOLD 0.05
#endif

/**
 * If there is an neighbor edge that has SMAA_LOCAL_CONTRAST_FACTOR times
 * bigger contrast than current edge, current edge will be discarded.
 *
 * This allows to eliminate spurious crossing edges, and is based on the fact
 * that, if there is too much contrast in a direction, that will hide
 * perceptually contrast in the other neighbors.
 */
#ifndef SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR
#define SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR 2.0
#endif

layout(std430) uniform SMAA_EDGE_DETECTION_FRAG_UBO_TYPE {
  vec4 SMAA_RT_METRICS;
  vec4 FULL_SMAA_RT_METRICS;
};

/**
 * Input texture
 */
uniform sampler2D colorTex;
layout(location = 0) in vec2 fTexCoord;

/**
 * output
 */
layout(location = 0) out vec4 color;

void SMAAEdgeDetectionCalcOffsets(vec2 texcoord,
                                  out vec4 offset[3]) {
    offset[0] = gl_FragCoord.xyxy + vec4(-1.0, 0.0, 0.0, 1.0);
    offset[0].x = max(offset[0].x, 0.0);
    offset[0].w = min(offset[0].w, SMAA_RT_METRICS.w - 1);
    offset[1] = gl_FragCoord.xyxy + vec4(1.0, 0.0, 0.0, -1.0);
    offset[1].x = min(offset[1].x, SMAA_RT_METRICS.z - 1);
    offset[1].w = max(offset[1].w, 0.0);
    offset[2] = gl_FragCoord.xyxy + vec4(-2.0, 0.0, 0.0, 2.0);
    offset[2].x = max(offset[2].x, 0.0);
    offset[2].w = min(offset[2].w, SMAA_RT_METRICS.w - 1);
}

/**
 * Gathers current pixel, and the top-left neighbors.
 */
vec3 SMAAGatherNeighbours(vec2 texcoord,
                          vec4 offset[3],
                          sampler2D tex) {
    return textureGather(tex, texcoord + FULL_SMAA_RT_METRICS.xy * vec2(-0.5, -0.5)).grb;
}

#ifdef SMAA_PREDICATION

/**
 * Threshold to be used in the additional predication buffer.
 *
 * Range: depends on the input, so you'll have to find the magic number that
 * works for you.
 */
#ifndef SMAA_PREDICATION_THRESHOLD
#define SMAA_PREDICATION_THRESHOLD 0.01
#endif

/**
 * How much to scale the global threshold used for luma or color edge
 * detection when using predication.
 *
 * Range: [1, 5]
 */
#ifndef SMAA_PREDICATION_SCALE
#define SMAA_PREDICATION_SCALE 2.0
#endif

/**
 * How much to locally decrease the threshold.
 *
 * Range: [0, 1]
 */
#ifndef SMAA_PREDICATION_STRENGTH
#define SMAA_PREDICATION_STRENGTH 0.4
#endif

uniform sampler2D predicationTex;

/**
 * Adjusts the threshold by means of predication.
 */
vec2 SMAACalculatePredicatedThreshold(vec2 texcoord,
                                      vec4 offset[3],
                                      sampler2D predicationTex) {
    vec3 neighbours = SMAAGatherNeighbours(texcoord, offset, predicationTex);
    vec2 delta = abs(neighbours.xx - neighbours.yz);
    vec2 edges = step(SMAA_PREDICATION_THRESHOLD, delta);
    return SMAA_PREDICATION_SCALE * SMAA_THRESHOLD * (1.0 - SMAA_PREDICATION_STRENGTH * edges);
}
#endif // SMAA_PREDICATION

 /**
 * Entry point: Luma Edge Detection
 *
 * IMPORTANT NOTICE: luma edge detection requires gamma-corrected colors, and
 * thus 'colorTex' should be a non-sRGB texture.
 */
void SMAALumaEdgeDetection() {
    vec4 offset[3];
    SMAAEdgeDetectionCalcOffsets(fTexCoord, offset);

    // Calculate the threshold:
    #ifdef SMAA_PREDICATION
    vec2 threshold = SMAACalculatePredicatedThreshold(fTexCoord, offset, predicationTex);
    #else
    vec2 threshold = vec2(SMAA_THRESHOLD, SMAA_THRESHOLD);
    #endif

    // Calculate lumas:
    vec3 weights = vec3(0.2126, 0.7152, 0.0722);
    vec4 C = texelFetch(colorTex, ivec2(gl_FragCoord.xy), 0);
    float L = dot(C.rgb, weights);

    vec4 Cleft = texelFetch(colorTex, ivec2(offset[0].xy), 0);
    float Lleft = dot(Cleft.rgb, weights);

    vec4 Ctop = texelFetch(colorTex, ivec2(offset[0].zw), 0);
    float Ltop  = dot(Ctop.rgb, weights);

    // We do the usual threshold:
    vec4 delta;
    delta.xy = abs(L - vec2(Lleft, Ltop));
    vec2 edges = step(threshold, delta.xy);

    // Then discard if there is no edge:
    if (dot(edges, vec2(1.0, 1.0)) == 0.0) {
        // check alpha differences before discarding
        // An edge can be determined via alpha when there's
        // a pixel with some alpha next to a pixel with no
        // alpha
        float alpha = step(1e-5, C.a);
        float alphaL = step(1e-5, Cleft.a);
        float alphaT = step(1e-5, Ctop.a);
        delta.x = abs(alphaL - alpha);
        delta.y = abs(alphaT - alpha);
        if (delta.x + delta.y == 0.0) {
            discard;
        }
        edges = delta.xy;
    }

    // Calculate right and bottom deltas:
    float Lright = dot(texelFetch(colorTex, ivec2(offset[1].xy), 0).rgb, weights);
    float Lbottom  = dot(texelFetch(colorTex, ivec2(offset[1].zw), 0).rgb, weights);
    delta.zw = abs(L - vec2(Lright, Lbottom));

    // Calculate the maximum delta in the direct neighborhood:
    vec2 maxDelta = max(delta.xy, delta.zw);

    // Calculate left-left and top-top deltas:
    float Lleftleft = dot(texelFetch(colorTex, ivec2(offset[2].xy), 0).rgb, weights);
    float Ltoptop = dot(texelFetch(colorTex, ivec2(offset[2].zw), 0).rgb, weights);
    delta.zw = abs(vec2(Lleft, Ltop) - vec2(Lleftleft, Ltoptop));

    // Calculate the final maximum delta:
    maxDelta = max(maxDelta.xy, delta.zw);
    float finalDelta = max(maxDelta.x, maxDelta.y);

    // Local contrast adaptation:
    edges.xy *= step(finalDelta, SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR * delta.xy);

    color = vec4(edges, 0.0, 0.0);
}

/**
 * Entry point: Color Edge Detection
 *
 * IMPORTANT NOTICE: color edge detection requires gamma-corrected colors, and
 * thus 'colorTex' should be a non-sRGB texture.
 */
void SMAAColorEdgeDetection() {
    vec4 offset[3];
    SMAAEdgeDetectionCalcOffsets(fTexCoord, offset);

    // Calculate the threshold:
    #ifdef SMAA_PREDICATION
    vec2 threshold = SMAACalculatePredicatedThreshold(fTexCoord, offset, predicationTex);
    #else
    vec2 threshold = vec2(SMAA_THRESHOLD, SMAA_THRESHOLD);
    #endif

    // Calculate color deltas:
    vec4 delta;
    vec4 C = texelFetch(colorTex, ivec2(gl_FragCoord.xy), 0);
    // an edge is only detected using alpha when
    // we go from some alpha to no alpha -- so do a step
    // on the alpha channel    C.a = step(1e-5, C.a);

    vec4 Cleft = texelFetch(colorTex, ivec2(offset[0].xy), 0);
    Cleft.a = step(1e-5, Cleft.a);
    vec4 t = abs(C - Cleft);
    delta.x = max(max(t.r, t.g), max(t.b, t.a));

    vec4 Ctop  = texelFetch(colorTex, ivec2(offset[0].zw), 0);
    Ctop.a = step(1e-5, Ctop.a);
    t = abs(C - Ctop);
    delta.y = max(max(t.r, t.g), max(t.b, t.a));

    // We do the usual threshold:
    vec2 edges = step(threshold, delta.xy);

    // Then discard if there is no edge:
    if (dot(edges, vec2(1.0, 1.0)) == 0.0)
        discard;

    // Calculate right and bottom deltas:
    vec4 Cright = texelFetch(colorTex, ivec2(offset[1].xy), 0);
    Cright.a = step(1e-5, Cright.a);
    t = abs(C - Cright);
    delta.z = max(max(t.r, t.g), max(t.b, t.a));

    vec4 Cbottom  = texelFetch(colorTex, ivec2(offset[1].zw), 0);
    Cbottom.a = step(1e-5, Cbottom.a);
    t = abs(C - Cbottom);
    delta.w = max(max(t.r, t.g), max(t.b, t.a));

    // Calculate the maximum delta in the direct neighborhood:
    vec2 maxDelta = max(delta.xy, delta.zw);

    // Calculate left-left and top-top deltas:
    vec4 Cleftleft  = texelFetch(colorTex, ivec2(offset[2].xy), 0);
    Cleftleft.a = step(1e-5, Cleftleft.a);
    t = abs(C - Cleftleft);
    delta.z = max(max(t.r, t.g), max(t.b, t.a));

    vec4 Ctoptop = texelFetch(colorTex, ivec2(offset[2].zw), 0);
    Ctoptop.a = step(1e-5, Ctoptop.a);
    t = abs(C - Ctoptop);
    delta.w = max(max(t.r, t.g), max(t.b, t.a));

    // Calculate the final maximum delta:
    maxDelta = max(maxDelta.xy, delta.zw);
    float finalDelta = max(maxDelta.x, maxDelta.y);

    // Local contrast adaptation:
    edges.xy *= step(finalDelta, SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR * delta.xy);

    color = vec4(edges, 0.0, 0.0);
}

/**
 * SMAA_DEPTH_THRESHOLD specifies the threshold for depth edge detection.
 *
 * Range: depends on the depth range of the scene.
 */
#ifndef SMAA_DEPTH_THRESHOLD
#define SMAA_DEPTH_THRESHOLD (0.1 * SMAA_THRESHOLD)
#endif

/**
 * Entry point: Depth Edge Detection
 */
void SMAADepthEdgeDetection() {
    vec4 offset[3];
    SMAAEdgeDetectionCalcOffsets(fTexCoord, offset);

    vec3 neighbours = SMAAGatherNeighbours(fTexCoord, offset, colorTex);
    vec2 delta = abs(neighbours.xx - vec2(neighbours.y, neighbours.z));
    vec2 edges = step(SMAA_DEPTH_THRESHOLD, delta);

    if (dot(edges, vec2(1.0, 1.0)) == 0.0)
        discard;

    color = vec4(edges, 0.0, 0.0);
}
