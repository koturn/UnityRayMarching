Shader "koturn/RayMarching/Sphere"
{
    Properties
    {
        _Color ("Color of the objects", Color) = (1.0, 1.0, 1.0, 1.0)
        _MaxLoop ("Maximum loop count", Int) = 60
        _MinRayDist ("Minimum distance of the ray", Float) = 0.01
        _MaxRayDist ("Maximum distance of the ray", Float) = 1000.0

        _SpecularPower ("Specular Power", Range(0.0, 100.0)) = 1.0
        _SpecularColor ("Specular Color", Color) = (0.5, 0.5, 0.5, 1.0)

        [KeywordEnum(Lembert, Half Lembert, Squred Half Lembert)]
        _DiffuseMode ("Reflection Model", Int) = 2

        [KeywordEnum(Original, Half Vector)]
        _SpecularMode ("Reflection Model", Int) = 1

        [KeywordEnum(Legacy, SH9)]
        _AmbientMode ("Reflection Model", Int) = 1

        [Toggle(_ENABLE_REFLECTION_PROBE)]
        _EnableReflectionProbe ("Enable Reflection Probe", Int) = 1

        _RefProbeBlendCoeff ("Blend coefficint of reflection probe", Range(0.0, 1.0)) = 0.5
    }

    SubShader
    {
        Tags
        {
            "RenderType" = "Opaque"
            "LightMode" = "ForwardBase"
        }

        Cull Front

        Pass
        {
            CGPROGRAM
            #pragma target 3.0
            #pragma vertex vert
            #pragma fragment frag

            #pragma shader_feature_local_fragment _DIFFUSEMODE_LEMBERT _DIFFUSEMODE_HALF_LEMBERT _DIFFUSEMODE_SQURED_HALF_LEMBERT
            #pragma shader_feature_local_fragment _SPECULARMODE_ORIGINAL _SPECULARMODE_HALF_VECTOR
            #pragma shader_feature_local_fragment _AMBIENTMODE_LEGACY _AMBIENTMODE_SH9
            #pragma shader_feature_local_fragment _ _ENABLE_REFLECTION_PROBE

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
            float sdSphere(float3 p, float r);
            float sdf(float3 p);
            float3 getNormal(float3 p);
            half4 getReflectionProbeColor(half3 refDir);


            //! Color of light.
            uniform fixed4 _LightColor0;

            //! Color of the objects.
            uniform float4 _Color;
            //! Maximum loop count.
            uniform int _MaxLoop;
            //! Minimum distance of the ray.
            uniform float _MinRayDist;
            //! Maximum distance of the ray.
            uniform float _MaxRayDist;
            //! Specular power.
            uniform float _SpecularPower;
            //! Specular color.
            uniform float4 _SpecularColor;
            //! Blend coefficint of reflection probe.
            uniform float _RefProbeBlendCoeff;


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

                // Distance.
                float d = 0.0;
                // Total distance.
                float t = 0.0;

                // Loop of Ray Marching.
                // UNITY_UNROLL
                for (int i = 0; i < _MaxLoop; i++) {
                    // Position of the tip of the ray.
                    const float3 p = localSpacaCameraPos + rayDir * t;
                    d = sdf(p);
                    t += d;
                    // Break this loop if the ray goes too far or collides.
                    if (d < _MinRayDist || t > _MaxRayDist) {
                        break;
                    }
                }

                // Discard if it is determined that the ray is not in collision.
                clip(_MinRayDist - d);

                const float3 finalPos = localSpacaCameraPos + rayDir * t;

                // Normal.
                const float3 normal = getNormal(finalPos);
                // Directional light angles to local coordinates.
                const float3 lightDir = normalize(mul(unity_WorldToObject, _WorldSpaceLightPos0).xyz);
                // View direction.
                const float3 viewDir = normalize(localSpacaCameraPos - finalPos);

                // Lambertian reflectance.
                const float3 lightCol = _LightColor0.rgb * LIGHT_ATTENUATION(i);
                const float nDotL = saturate(dot(normal, lightDir));
#if defined(_DIFFUSEMODE_SQURED_HALF_LEMBERT)
                const float diffuse = lightCol * sq(nDotL * 0.5 + 0.5);
#elif defined(_DIFFUSEMODE_HALF_LEMBERT)
                const float3 diffuse = lightCol * (nDotL * 0.5 + 0.5);
#else
                const float3 diffuse = lightCol * nDotL;
#endif  // defined(_DIFFUSEMODE_SQURED_HALF_LEMBERT)

                // Specular reflection.
#ifdef _SPECULARMODE_HALF_VECTOR
                const float3 specular = pow(max(0.0, dot(normalize(lightDir + viewDir), normal)), _SpecularPower) * _SpecularColor.xyz * lightCol;
#else
                const float3 specular = pow(max(0.0, dot(reflect(-lightDir, normal), viewDir)), _SpecularPower) * _SpecularColor.xyz * lightCol;
#endif  // _SPECULARMODE_HALF_VECTOR

                // Ambient color.
#ifdef _AMBIENTMODE_SH9
                const float3 ambient = ShadeSH9(half4(UnityObjectToWorldNormal(normal), 1.0));
#else
                const float3 ambient = UNITY_LIGHTMODEL_AMBIENT.rgb;
#endif  // _AMBIENTMODE_SH9

#ifdef _ENABLE_REFLECTION_PROBE
                const half3 refDir = UnityObjectToWorldNormal(reflect(-viewDir, normal));
                const half4 refColor = getReflectionProbeColor(refDir);
                const float4 col = float4((diffuse + ambient) * lerp(_Color.rgb, refColor.rgb, _RefProbeBlendCoeff) + specular, _Color.a);
#else
                // Output color.
                const float4 col = float4((diffuse + ambient) * _Color.rgb + specular, _Color.a);
#endif  // _ENABLE_REFLECTION_PROBE

                pout o;
                o.color = col;
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
             * @brief SDF of Sphere.
             * @param [in] r  Radius of sphere.
             * @return Signed Distance to the Sphere.
             */
            float sdSphere(float3 p, float r)
            {
                return length(p) - r;
            }

            /*!
             * @brief SDF (Signed Distance Function) of objects.
             * @param [in] p  Position of the tip of the ray.
             * @return Signed Distance to the objects.
             */
            float sdf(float3 p)
            {
                return sdSphere(p, 0.5);
            }

            /*!
             * @brief Calculate normal of the objects.
             * @param [in] p  Position of the tip of the ray.
             * @return Normal of the objects.
             */
            float3 getNormal(float3 p)
            {
#if 1
                // Lightweight normal calculation.
                // Distance function calling is just four.
                // See: https://iquilezles.org/articles/normalsSDF/
                static const float h = 0.0001;
                static const float2 k = float2(1.0, -1.0);

                return normalize(
                    k.xyy * sdf(p + k.xyy * h)
                        + k.yyx * sdf(p + k.yyx * h)
                        + k.yxy * sdf(p + k.yxy * h)
                        + k.xxx * sdf(p + k.xxx * h));

                // static const float2 k = float2(1.0, -1.0);
                // static const float2 kh = k * 0.0001;

                // return normalize(
                //     k.xyy * sdf(p + kh.xyy)
                //         + k.yyx * sdf(p + kh.yyx)
                //         + k.yxy * sdf(p + kh.yxy)
                //         + k.xxx * sdf(p + kh.xxx));

#else
                // Naive normal calculation.
                // Distance function calling is six.
                // static const float d = 0.0001;

                // return normalize(
                //     float3(
                //         sdf(p + float3(d, 0.0, 0.0)) - sdf(p + float3(-d, 0.0, 0.0)),
                //         sdf(p + float3(0.0, d, 0.0)) - sdf(p + float3(0.0, -d, 0.0)),
                //         sdf(p + float3(0.0, 0.0, d)) - sdf(p + float3(0.0, 0.0, -d))));

                static const float2 d = float2(0.0001, 0.0)

                return normalize(
                    float3(
                        sdf(p + d.xyy) - sdf(p - d.xyy),
                        sdf(p + d.yxy) - sdf(p - d.yxy),
                        sdf(p + d.yyx) - sdf(p - d.yyx)));
#endif
            }

            /*!
             * @brief Get color of reflection probe.
             * @param [in] refDir  Reflect direction.
             * @return Color of reflection probe.
             */
            half4 getReflectionProbeColor(half3 refDir)
            {
                half4 refColor = UNITY_SAMPLE_TEXCUBE_LOD(unity_SpecCube0, refDir, 0.0);
                refColor.rgb = DecodeHDR(refColor, unity_SpecCube0_HDR);
                return refColor;
            }
            ENDCG
        }
    }
}
