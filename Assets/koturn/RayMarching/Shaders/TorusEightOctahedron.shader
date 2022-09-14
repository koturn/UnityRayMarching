Shader "koturn/RayMarching/TorusEightOctahedron"
{
    Properties
    {
        // Common Ray Marching Parameters.
        _MaxLoop ("Maximum loop count", Int) = 256
        _MinRayDist ("Minimum distance of the ray", Float) = 0.001
        _MaxRayDist ("Maximum distance of the ray", Float) = 1000.0
        _Scale ("Scale vector", Vector) = (1.0, 1.0, 1.0, 1.0)
        _MarchingFactor ("Marching Factor", Float) = 0.5

        _SpecularPower ("Specular Power", Range(0.0, 100.0)) = 5.0
        _SpecularColor ("Color of specular", Color) = (0.5, 0.5, 0.5, 0.5)

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
            float sdf(float3 p, out float3 color);
            float3 getNormal(float3 p);
            float sdTorus(float3 p, float2 t);
            float sdOctahedron(float3 p, float s);
            float2 rotate2D(float2 v, float angle);


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

                float3 color = float3(0.0, 0.0, 0.0);

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
                const float3 specular = pow(max(0.0, dot(normalize(lightDir + viewDir), normal)), _SpecularPower) * _SpecularColor.xyz * lightCol;

                // Ambient color.
                const float3 ambient = ShadeSH9(half4(UnityObjectToWorldNormal(normal), 1.0));

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
            float sdf(float3 p, out float3 color)
            {
                static const float3 kColors[8] = {
                    float3(0.4, 0.8, 0.4),
                    float3(0.8, 0.4, 0.4),
                    float3(0.8, 0.8, 0.8),
                    float3(0.8, 0.8, 0.4),
                    float3(0.8, 0.4, 0.8),
                    float3(0.4, 0.8, 0.8),
                    float3(0.4, 0.4, 0.4),
                    float3(0.4, 0.4, 0.8)
                };
                static const float kQuarterPi = UNITY_PI / 4.0;
                static const float kInvQuarterPi = 1.0 / kQuarterPi;

                const float radius = _TorusRadius + _SinTime.w * _TorusRadiusAmp;

                float d = sdTorus(p, float2(radius, _TorusWidth));
                color = float3(1.0, 1.0, 1.0);

                p.xy = rotate2D(p.xy, _Time.y);
                const float xyAngle = atan2(p.y, p.x);
                const float rotUnit = floor(xyAngle * kInvQuarterPi);
                p.xy = rotate2D(p.xy, kQuarterPi * rotUnit + kQuarterPi / 2.0);

                const float nd = sdOctahedron(p - float3(radius, 0.0, 0.0), _OctahedronSize);
                if (d > nd) {
                    d = nd;
                    const int idx = (int)rotUnit;
                    color = idx == 0 ? kColors[0]
                        : idx == 1 ? kColors[1]
                        : idx == 2 ? kColors[2]
                        : idx == 3 ? kColors[3]
                        : idx == -4 ? kColors[4]
                        : idx == -3 ? kColors[5]
                        : idx == -2 ? kColors[6]
                        : kColors[7];
                }

                return d;
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

                float3 _ = float3(0.0, 0.0, 0.0);
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
            float sdOctahedron(float3 p, float s)
            {
                return (dot(abs(p), float3(0.5, 2.0, 2.0)) - s) * 0.57735027;
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
            ENDCG
        }
    }
}
