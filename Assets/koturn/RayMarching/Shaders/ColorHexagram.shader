Shader "koturn/RayMarching/ColorHexagram"
{
    Properties
    {
        // Common Ray Marching Parameters.
        _MaxLoop ("Maximum loop count", Int) = 256
        _MinRayDist ("Minimum distance of the ray", Float) = 0.001
        _MaxRayDist ("Maximum distance of the ray", Float) = 1000.0
        _Scale ("Scale vector", Vector) = (1.0, 1.0, 1.0, 1.0)
        _MarchingFactor ("Marching Factor", Float) = 0.5

        // Lighting Parameters.
        _SpecularPower ("Specular Power", Range(0.0, 100.0)) = 1.0
        _SpecularColor ("Color of specular", Color) = (1.0, 1.0, 1.0, 1.0)
        _LineMultiplier ("Multiplier of lines", Float) = 5.0

        // SDF parameters.
        _TorusRadius ("Radius of Torus", Float) = 0.25
        _TorusRadiusAmp ("Radius Amplitude of Torus", Float) = 0.05
        _TorusWidth ("Width of Torus", Float) = 0.005
        _OctahedronSize ("Size of Octahedron", Float) = 0.05

        [Enum(UnityEngine.Rendering.CullMode)]
        _Cull ("Culling Mode", Int) = 1  // Default: Front

        [Enum(Off, 0, On, 1)]
        _AlphaToMask ("Alpha To Mask", Int) = 0  // Default: Off
    }

    SubShader
    {
        Tags
        {
            "RenderType" = "Opaque"
            "LightMode" = "ForwardBase"
            "IgnoreProjector" = "True"
            "VRCFallback" = "Hidden"
        }

        Cull [_Cull]
        AlphaToMask [_AlphaToMask]

        Pass
        {
            CGPROGRAM
            #pragma target 3.0
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"
            #include "AutoLight.cginc"


            /*!
             * @brief Input of vertex shader.
             */
            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            /*!
             * @brief Output of vertex shader and input of fragment shader.
             */
            struct v2f
            {
                float2 uv : TEXCOORD0;
                float3 pos : TEXCOORD1;
                float4 vertex : SV_POSITION;
            };

            /*!
             * @brief Output of fragment shader.
             */
            struct pout
            {
                fixed4 color : SV_Target;
                float depth : SV_Depth;
            };


            float sq(float x);
            float sdf(float3 p, out float4 color);
            float3 getNormal(float3 p);
            float sdTorus(float3 p, float2 t);
            float sdOctahedron(float3 p, float3 scale, float s);
            float sdCappedCylinder(float3 p, float h, float r);
            float2 rotate2D(float2 v, float2 pivot, float angle);
            float2 rotate2D(float2 v, float angle);
            float3 rgb2hsv(float3 rgb);
            float3 hsv2rgb(float3 hsv);
            float3 rgbAddHue(float3 rgb, float hue);


            //! Color of light.
            uniform fixed4 _LightColor0;

            //! Maximum loop count.
            uniform int _MaxLoop;
            //! Minimum distance of the ray.
            uniform float _MinRayDist;
            //! Maximum distance of the ray.
            uniform float _MaxRayDist;
            //! Scale vector.
            uniform float3 _Scale;
            //! Marching Factor.
            uniform float _MarchingFactor;

            //! Power of specular.
            uniform float _SpecularPower;
            //! Color of specular.
            uniform float4 _SpecularColor;
            //! Multiplier of lines.
            uniform float _LineMultiplier;

            //! Radius of Torus.
            uniform float _TorusRadius;
            //! Radius Amplitude of Torus.
            uniform float _TorusRadiusAmp;
            //! Width of Torus.
            uniform float _TorusWidth;
            //! Size of Octahedron.
            uniform float _OctahedronSize;


            /*!
             * @brief Vertex shader function.
             * @param [in] v  Input data
             * @return Output for fragment shader (v2f).
             */
            v2f vert(appdata v)
            {
                v2f o;

                o.vertex = UnityObjectToClipPos(v.vertex);
                o.pos = v.vertex.xyz;
                o.uv = v.uv;

                return o;
            }


            /*!
             * @brief Fragment shader function.
             * @param [in] i  Input data from vertex shader
             * @return Output of each texels (pout).
             */
            pout frag(v2f i)
            {
                // The start position of the ray is the local coordinate of the camera.
                const float3 localSpacaCameraPos = mul(unity_WorldToObject, float4(_WorldSpaceCameraPos, 1.0)).xyz;
                // Define ray direction by finding the direction of the local coordinates
                // of the mesh from the local coordinates of the viewpoint.
                const float3 rayDir = normalize(i.pos.xyz - localSpacaCameraPos);

                const float invScale = 1.0 / _Scale;
                // Distance.
                float d = 0.0;
                // Total distance.
                float t = 0.0;

                float4 color = float4(0.0, 0.0, 0.0, 0.0);

                // Loop of Ray Marching.
                for (int i = 0; i < _MaxLoop; i++) {
                    // Position of the tip of the ray.
                    const float3 p = localSpacaCameraPos + rayDir * t;
                    d = sdf(p * _Scale, color) * invScale;

                    // Break this loop if the ray goes too far or collides.
                    if (d < _MinRayDist) {
                        break;
                    }

                    t += d * _MarchingFactor;

                    // Discard if the ray goes too far or collides.
                    clip(_MaxRayDist - t);
                }

                // Discard if it is determined that the ray is not in collision.
                clip(_MinRayDist - d);

                const float3 finalPos = localSpacaCameraPos + rayDir * t;

                const float3 normal = getNormal(finalPos);
                // Directional light angles to local coordinates.
                const float3 lightDir = normalize(mul(unity_WorldToObject, _WorldSpaceLightPos0).xyz);
                // View direction.
                const float3 viewDir = normalize(localSpacaCameraPos - finalPos);

                // Lambertian reflectance: Half-Lambert.
                const float3 lightCol = _LightColor0.rgb * LIGHT_ATTENUATION(i);
                const float nDotL = saturate(dot(normal, lightDir));
                const float3 diffuse = lightCol * sq(nDotL * 0.5 + 0.5);

                // Specular reflection.
                const float3 specular = color.a * pow(max(0.0, dot(normalize(lightDir + viewDir), normal)), _SpecularPower) * _SpecularColor.xyz * lightCol;

                // Ambient color.
                const float3 ambient = ShadeSH9(half4(normal, 1.0));

                pout o;
                o.color = float4((diffuse + ambient) * color.rgb + specular, 1.0);
                const float4 projectionPos = UnityObjectToClipPos(float4(finalPos, 1.0));
                o.depth = projectionPos.z / projectionPos.w;

                return o;
            }


            /*!
             * @brief Calculate squared value.
             * @param [in] x  A value.
             * @return x * x
             */
            float sq(float x)
            {
                return x * x;
            }

            /*!
             * @brief SDF (Signed Distance Function) of objects.
             * @param [in] p  Position of the tip of the ray.
             * @return Signed Distance to the objects.
             */
            float sdf(float3 p, out float4 color)
            {
                static const float4 kColors[6] = {
                    float4(0.8, 0.4, 0.4, 1.0),  // R
                    float4(0.8, 0.8, 0.4, 1.0),  // Y
                    float4(0.4, 0.8, 0.4, 1.0),  // G
                    float4(0.4, 0.8, 0.8, 1.0),  // C
                    float4(0.4, 0.4, 0.8, 1.0),  // B
                    float4(0.8, 0.4, 0.8, 1.0)   // M
                };
                static const float kOneThirdPi = UNITY_PI / 3.0;
                static const float kTwoThirdPi = UNITY_PI * (2.0 / 3.0);
                static const float kOneSixthPi = UNITY_PI / 6.0;
                static const float kInvOneThirdPi = rcp(kOneThirdPi);
                static const float kInvTwoThirdPi = rcp(kTwoThirdPi);

                const float radius = _TorusRadius + _SinTime.w * _TorusRadiusAmp;

                float minDist = sdTorus(p, float2(radius, _TorusWidth));

                p.xy = rotate2D(p.xy, _Time.y);

                const float xyAngle = atan2(p.y, p.x);
                color = float4(
                    rgbAddHue(float3(1.0, 0.75, 0.25), xyAngle / UNITY_TWO_PI + rcp(UNITY_PI / 12.0)) * _LineMultiplier,
                    0.0);

                const float rotUnit = floor(xyAngle * kInvOneThirdPi);
                float3 rayPos1 = p;
                rayPos1.xy = rotate2D(rayPos1.xy, kOneThirdPi * rotUnit + kOneSixthPi);

                const float dist = sdOctahedron(rayPos1 - float3(radius, 0.0, 0.0), float3(2.0, 2.0, 0.5), _OctahedronSize);
                if (minDist > dist) {
                    minDist = dist;
                    const int idx = ((int)rotUnit);
                    color = idx == 0 ? kColors[0]
                        : idx == 1 ? kColors[1]
                        : idx == 2 ? kColors[2]
                        : idx == -3 ? kColors[3]
                        : idx == -2 ? kColors[4]
                        : kColors[5];
                }

                const float2 posXY1 = rotate2D(float2(radius, 0.0), kTwoThirdPi);
                const float2 posXY2 = rotate2D(float2(radius, 0.0), -kTwoThirdPi);
                const float2 posCenterXY = (posXY1 + posXY2) * 0.5;
                const float length12 = length(posXY2 - posXY1) * 0.5;
#if 0
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 3; j++) {
                        float3 rayPos2 = p;
                        rayPos2.xy = rotate2D(rayPos2.xy, kTwoThirdPi * j + kOneThirdPi * (i + 3) + kOneSixthPi);
                        rayPos2.xy -= rotate2D(posCenterXY, posCenterXY, kTwoThirdPi * j + kOneSixthPi);
                        const float dist2 = sdCappedCylinder(rayPos2, 0.0025, length12);
                        if (minDist > dist2) {
                            minDist = dist2;
                            color = float4(kColors[j * 2 + i].xyz * _LineMultiplier, 0.0);
                        }
                    }
                }
#else
                for (int i = 0; i < 2; i++) {
                    const float rotUnit2 = floor((xyAngle + kOneSixthPi - kOneThirdPi * i) * kInvTwoThirdPi);

                    float3 rayPos2 = p;
                    rayPos2.xy = rotate2D(rayPos2.xy, kTwoThirdPi * rotUnit2 + kOneThirdPi * (i + 3) + kOneSixthPi);
                    rayPos2.xy -= rotate2D(posCenterXY, posCenterXY, kTwoThirdPi * rotUnit2 + kOneSixthPi);

                    const float dist2 = sdCappedCylinder(rayPos2, 0.0025, length12 * 5);
                    if (minDist > dist2) {
                        minDist = dist2;
                        const int idx = int(rotUnit2);
                        const float3 albedo = idx == 0 ? kColors[0 + i]
                            : idx == -1 ? kColors[4 + i]
                            : kColors[2 + i];
                        color = float4(albedo * _LineMultiplier, 0.0);
                    }
                }
#endif

                return minDist;
            }

            /*!
             * @brief Calculate normal of the objects.
             * @param [in] p  Position of the tip of the ray.
             * @return Normal of the objects.
             */
            float3 getNormal(float3 p)
            {
                // See: https://iquilezles.org/articles/normalsSDF/
                static const float2 k = float2(1.0, -1.0);
                static const float2 kh = k * 0.0001;

                float4 _ = float4(0.0, 0.0, 0.0, 0.0);
                return normalize(
                    k.xyy * sdf(p + kh.xyy, _)
                        + k.yyx * sdf(p + kh.yyx, _)
                        + k.yxy * sdf(p + kh.yxy, _)
                        + k.xxx * sdf(p + kh.xxx, _));
            }

            /*!
             * @brief SDF of Torus.
             * @param [in] p  Position of the tip of the ray.
             * @param [in] t  (t.x, t.y) = (radius of torus, thickness of torus).
             * @return Signed Distance to the Sphere.
             */
            float sdTorus(float3 p, float2 t)
            {
                const float2 q = float2(length(p.xy) - t.x, p.z);
                return length(q) - t.y;
            }

            /*!
             * @brief SDF of Octahedron.
             * @param [in] p  Position of the tip of the ray.
             * @param [in] s  Size of Octahedron.
             * @return Signed Distance to the Octahedron.
             */
            float sdOctahedron(float3 p, float3 scale, float s)
            {
                return (dot(abs(p), scale) - s) * 0.57735027;
            }

            float sdCappedCylinder(float3 p, float h, float r)
            {
                const float2 d = abs(float2(length(p.xz), p.y)) - float2(h, r);
                return min(0.0, max(d.x, d.y)) + length(max(d, 0.0));
            }

            /*!
             * @brief Rotate on 2D plane
             * @param [in] v  Target vector
             * @param [in] pivot  Pivot of rotation.
             * @param [in] angle  Angle of rotation.
             * @return Rotated vector.
             */
            float2 rotate2D(float2 v, float2 pivot, float angle)
            {
                return rotate2D(v - pivot, angle) + pivot;
            }

            /*!
             * @brief Rotate on 2D plane
             * @param [in] v  Target vector
             * @param [in] angle  Angle of rotation.
             * @return Rotated vector.
             */
            float2 rotate2D(float2 v, float angle)
            {
                float s, c;
                sincos(angle, s, c);
                return float2(dot(v, float2(c, s)), dot(v, float2(-s, c)));
            }

            /*!
             * @brief Convert from RGB to HSV.
             *
             * @param [in] rgb  Three-dimensional vector of RGB.
             * @return Three-dimensional vector of HSV.
             */
            float3 rgb2hsv(float3 rgb)
            {
                static const float4 k = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
                static const float e = 1.0e-10;

                const float4 p = rgb.g < rgb.b ? float4(rgb.bg, k.wz) : float4(rgb.gb, k.xy);
                const float4 q = rgb.r < p.x ? float4(p.xyw, rgb.r) : float4(rgb.r, p.yzx);
                const float d = q.x - min(q.w, q.y);
                return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
            }

            /*!
             * @brief Convert from HSV to RGB.
             *
             * @param [in] hsv  Three-dimensional vector of HSV.
             * @return Three-dimensional vector of RGB.
             */
            float3 hsv2rgb(float3 hsv)
            {
                static const float4 k = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);

                const float3 p = abs(frac(hsv.xxx + k.xyz) * 6.0 - k.www);
                return hsv.z * lerp(k.xxx, saturate(p - k.xxx), hsv.y);
            }

            /*!
             * @brief Add hue to RGB color.
             *
             * @param [in] rgb  Three-dimensional vector of RGB.
             * @param [in] hue  Scalar of hue.
             * @return Three-dimensional vector of RGB.
             */
            float3 rgbAddHue(float3 rgb, float hue)
            {
                float3 hsv = rgb2hsv(rgb);
                hsv.x += hue;
                return hsv2rgb(hsv);
            }
            ENDCG
        }
    }
}
