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

//-----------------------------------------------------------------------------
// Texture Access Defines

#ifndef SMAA_DECODE_VELOCITY
#define SMAA_DECODE_VELOCITY(sample) sample.rg
#endif

//-----------------------------------------------------------------------------
// Global Uniforms

layout(std430) uniform SMAA_NEIGHBORHOOD_BLENDING_FRAG_UBO_TYPE {
  vec4 SMAA_RT_METRICS;
  vec4 FULL_SMAA_RT_METRICS;
  float sampleWeight;
};

/**
 * Conditional move:
 */
void SMAAMovc(bvec2 cond, inout vec2 variable, vec2 value) {
    if (cond.x) variable.x = value.x;
    if (cond.y) variable.y = value.y;
}

void SMAAMovc(bvec4 cond, inout vec4 variable, vec4 value) {
    SMAAMovc(cond.xy, variable.xy, value.xy);
    SMAAMovc(cond.zw, variable.zw, value.zw);
}

//-----------------------------------------------------------------------------
// Neighborhood Blending Pixel Shader (Third Pass)

vec4 SMAANeighborhoodBlending(vec2 texcoord,
                                vec4 offset,
                                sampler2D colorTex,
                                sampler2D blendTex
                                #ifdef SMAA_REPROJECTION
                                , sampler2D velocityTex
                                #endif
                                ) {
    // Fetch the blending weights for current pixel:
    vec4 a;
    a.x = texture(blendTex, offset.xy).a; // Right
    a.y = texture(blendTex, offset.zw).g; // Top
    a.wz = texture(blendTex, texcoord).rb; // Bottom / Left

    // Is there any blending weight with a value greater than 0.0?

    if (dot(a, vec4(1.0, 1.0, 1.0, 1.0)) < 1e-5) {
        vec4 color = textureLod(colorTex, texcoord, 0.0);

        #ifdef SMAA_REPROJECTION
        vec2 velocity = SMAA_DECODE_VELOCITY(textureLod(velocityTex, texcoord, 0.0));

        // Pack velocity into the alpha channel:
        color.a = sqrt(5.0 * length(velocity));
        #endif

        return color;
    } else {
        bool h = max(a.x, a.z) > max(a.y, a.w); // max(horizontal) > max(vertical)

        // Calculate the blending offsets:
        vec4 blendingOffset = vec4(0.0, -a.y, 0.0, -a.w);
        vec2 blendingWeight = a.yw;
        SMAAMovc(bvec4(h, h, h, h), blendingOffset, vec4(a.x, 0.0, a.z, 0.0));
        SMAAMovc(bvec2(h, h), blendingWeight, a.xz);
        blendingWeight /= dot(blendingWeight, vec2(1.0, 1.0));

        // Calculate the texture coordinates:
        vec4 blendingCoord = fma(blendingOffset, vec4(FULL_SMAA_RT_METRICS.xy, -FULL_SMAA_RT_METRICS.xy), texcoord.xyxy);

        // We exploit bilinear filtering to mix current pixel with the chosen
        // neighbor:
        vec4 color = blendingWeight.x * textureLod(colorTex, blendingCoord.xy, 0.0);
        color += blendingWeight.y * textureLod(colorTex, blendingCoord.zw, 0.0);

        #ifdef SMAA_REPROJECTION
        // Antialias velocity for proper reprojection in a later stage:
        vec2 velocity = blendingWeight.x * SMAA_DECODE_VELOCITY(textureLod(velocityTex, blendingCoord.xy, 0.0));
        velocity += blendingWeight.y * SMAA_DECODE_VELOCITY(textureLod(velocityTex, blendingCoord.zw, 0.0));

        // Pack velocity into the alpha channel:
        color.a = sqrt(5.0 * length(velocity));
        #endif

        return color;
    }
}

/**
 * Neighborhood Blending
 */
void SMAANeighborhoodBlendingOffset(vec2 texcoord,
                                    out vec4 offset) {
    offset = fma(FULL_SMAA_RT_METRICS.xyxy, vec4( 1.0, 0.0, 0.0, -1.0), texcoord.xyxy);
}

uniform sampler2D colorTex;
uniform sampler2D blendTex;

#ifdef SMAA_REPROJECTION
uniform sampler2D velocityTex;
#endif

layout(location = 0) in vec2 fTexCoord;

layout(location = 0) out vec4 color;

void main(void)
{
    vec4 offset = vec4(0.0, 0.0, 0.0, 0.0);
    SMAANeighborhoodBlendingOffset(fTexCoord, offset);

    color = SMAANeighborhoodBlending(fTexCoord,
                                     offset,
                                     colorTex,
                                     blendTex
                                     #ifdef SMAA_REPROJECTION
                                     , velocityTex
                                     #endif
                                     ) * sampleWeight;
}
