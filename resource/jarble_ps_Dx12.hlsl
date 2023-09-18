// pixel shader
cbuffer PerFrameConstants : register (b0) { float2 iResolution; }
cbuffer PerFrameConstants : register (b1) { float iTime; }
cbuffer PerFrameConstants : register (b2) { uint2 iMouse; }

#define DRAG_MULT 0.048
#define ITERATIONS_RAYMARCH 13
// If set 1, then arti is visible. Original is 48
#define ITERATIONS_NORMAL 48

#define Mouse (iMouse.xy / iResolution.xy) // OLD
//static const float2 Mouse = float2( 1.7,0.5 ); // more artif, invisible light source
//static const float2 Mouse = float2( 1.55, 0.55 ); // less artif, visible light source

// NEW
float oct(float p){
    return frac(4768.1232345456 * sin(p));
}
float oct(float2 p){
    return frac(4768.1232345456 * sin((p.x+p.y*43.0)));
}
float oct(float3 p){
    return frac(4768.1232345456 * sin((p.x+p.y*43.0+p.z*137.0)));
}
float oct(float4 p){
    return frac(4768.1232345456 * sin((p.x+p.y*43.0+p.z*137.0+p.w*2666.0)));
}
float achnoise(float4 x){
    float4 p = floor(x);
    float4 fr = frac(x);
    float4 LBZU = p + float4(0.0, 0.0, 0.0, 0.0);
    float4 LTZU = p + float4(0.0, 1.0, 0.0, 0.0);
    float4 RBZU = p + float4(1.0, 0.0, 0.0, 0.0);
    float4 RTZU = p + float4(1.0, 1.0, 0.0, 0.0);

    float4 LBFU = p + float4(0.0, 0.0, 1.0, 0.0);
    float4 LTFU = p + float4(0.0, 1.0, 1.0, 0.0);
    float4 RBFU = p + float4(1.0, 0.0, 1.0, 0.0);
    float4 RTFU = p + float4(1.0, 1.0, 1.0, 0.0);

    float4 LBZD = p + float4(0.0, 0.0, 0.0, 1.0);
    float4 LTZD = p + float4(0.0, 1.0, 0.0, 1.0);
    float4 RBZD = p + float4(1.0, 0.0, 0.0, 1.0);
    float4 RTZD = p + float4(1.0, 1.0, 0.0, 1.0);

    float4 LBFD = p + float4(0.0, 0.0, 1.0, 1.0);
    float4 LTFD = p + float4(0.0, 1.0, 1.0, 1.0);
    float4 RBFD = p + float4(1.0, 0.0, 1.0, 1.0);
    float4 RTFD = p + float4(1.0, 1.0, 1.0, 1.0);

    float l0candidate1  = oct(LBZU);
    float l0candidate2  = oct(RBZU);
    float l0candidate3  = oct(LTZU);
    float l0candidate4  = oct(RTZU);

    float l0candidate5  = oct(LBFU);
    float l0candidate6  = oct(RBFU);
    float l0candidate7  = oct(LTFU);
    float l0candidate8  = oct(RTFU);

    float l0candidate9  = oct(LBZD);
    float l0candidate10 = oct(RBZD);
    float l0candidate11 = oct(LTZD);
    float l0candidate12 = oct(RTZD);

    float l0candidate13 = oct(LBFD);
    float l0candidate14 = oct(RBFD);
    float l0candidate15 = oct(LTFD);
    float l0candidate16 = oct(RTFD);

    float l1candidate1 = lerp(l0candidate1, l0candidate2, fr[0]);
    float l1candidate2 = lerp(l0candidate3, l0candidate4, fr[0]);
    float l1candidate3 = lerp(l0candidate5, l0candidate6, fr[0]);
    float l1candidate4 = lerp(l0candidate7, l0candidate8, fr[0]);
    float l1candidate5 = lerp(l0candidate9, l0candidate10, fr[0]);
    float l1candidate6 = lerp(l0candidate11, l0candidate12, fr[0]);
    float l1candidate7 = lerp(l0candidate13, l0candidate14, fr[0]);
    float l1candidate8 = lerp(l0candidate15, l0candidate16, fr[0]);


    float l2candidate1 = lerp(l1candidate1, l1candidate2, fr[1]);
    float l2candidate2 = lerp(l1candidate3, l1candidate4, fr[1]);
    float l2candidate3 = lerp(l1candidate5, l1candidate6, fr[1]);
    float l2candidate4 = lerp(l1candidate7, l1candidate8, fr[1]);


    float l3candidate1 = lerp(l2candidate1, l2candidate2, fr[2]);
    float l3candidate2 = lerp(l2candidate3, l2candidate4, fr[2]);

    float l4candidate1 = lerp(l3candidate1, l3candidate2, fr[3]);

    return l4candidate1;
}
#define noise3d(a) achnoise(a)
float supernoise3dX(float3 p){
	float a =  noise3d( float4( p, 0 ) );
	float b =  noise3d( float4( p + 10.5, 0 ) );
	return (a * b);
}
// NEW


float2 wavedx(float2 position, float2 direction, float speed, float frequency, float timeshift) {
    float x = dot(direction, position) * frequency + timeshift * speed;
    float wave = exp(sin(x) - 1.0);
    float dx = wave * cos(x);
    return float2(wave, -dx);
}

float getwaves(float2 position, int iterations){
	float iter = 0.0;
    float phase = 6.0;
    float speed = 2.0;
    float weight = 1.0;
    float w = 0.0;
    float ws = 0.0;
    for(int i=0;i<iterations;i++){
        float2 p = float2(sin(iter), cos(iter));
        float2 res = wavedx(position, p, speed, phase, iTime);
        position += p * res.y * weight * DRAG_MULT;
        w += res.x * weight;
        iter += 12.0;
        ws += weight;
        //weight = mix(weight, 0.0, 0.2); // OLD
        weight = lerp(weight, 0.0, 0.2); // NEW
        phase *= 1.18;
        speed *= 1.07;
    }
    return w / ws
		// But waves will most little
		// NEW
		//* supernoise3dX(0.3 *float3(position.x, position.y, 0.0) + iTime * 0.1) 
	;
}

float raymarchwater(float3 camera, float3 start, float3 end, float depth){
    float3 pos = start;
    float h = 0.0;
    float hupper = depth;
    float hlower = 0.0;
    float2 zer = float2( 0.0, 0.0 );
    float3 dir = normalize(end - start);
    float eps = 0.01;
    for(int i=0;i<318;i++){ 
        h = getwaves(pos.xz * 0.1, ITERATIONS_RAYMARCH) * depth - depth;
        float dist_pos = distance(pos, camera);
        if(h + eps*dist_pos > pos.y) {
            return dist_pos;
        }
        pos += dir * (pos.y - h);
        //eps *= 1.01;
    }
    return -1.0;
}

float3 normal(float2 pos, float e, float depth){
    float2 ex = float2(e, 0);
    float H = getwaves(pos.xy * 0.1, ITERATIONS_NORMAL) * depth;
    float3 a = float3(pos.x, H, pos.y);
    return (cross(normalize(a-float3(pos.x - e, getwaves((pos.xy - ex.xy)*0.1, ITERATIONS_NORMAL) * depth, pos.y)), 
                           normalize(a-float3(pos.x, getwaves((pos.xy + ex.yx )* 0.1, ITERATIONS_NORMAL) * depth, pos.y + e))));
}
float3x3 rotmat(float3 axis, float angle)
{
	axis = (axis);
	float s = sin(angle);
	float c = cos(angle);
	float oc = 1.0 - c;
	//return float3x3(oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s, // OLD
	return float3x3(oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s, 
	oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s, 
	oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c);
}

float3 getRay(float2 uv){
    uv = (uv * 2.0 - 1.0) * float2(iResolution.x / iResolution.y, 1.0);
	float3 proj = normalize(float3(uv.x, uv.y, 1.0) + float3(uv.x, uv.y, -1.0) * pow(length(uv), 2.0) * 0.05);	
    if(iResolution.x < 400.0) 
		return proj;
	//float3 ray = rotmat(float3(0.0, -1.0, 0.0), 3.0 * (Mouse.x * 2.0 - 1.0)) * rotmat(float3(1.0, 0.0, 0.0), 1.5 * (Mouse.y * 2.0 - 1.0)) * proj; // OLD
	float3x3 rotmat1 = 
		rotmat(
				float3(0.0, -1.0, 0.0), 3.0 * (Mouse.x * 2.0 - 1.0)) * rotmat(float3(1.0, 0.0, 0.0)
				, 1.5 * (Mouse.y * 2.0 - 1.0)
			);
	float3 rotmat2 = mul( proj, rotmat1 );
	float3 ray = rotmat2;
    return ray;
}

float intersectPlane(float3 origin, float3 direction, float3 point_, float3 normal)
{ 
    return clamp(dot(point_ - origin, normal) / dot(direction, normal), -1.0, 9991999.0); 
}

float3 extra_cheap_atmosphere(float3 raydir, float3 sundir){
	sundir.y = max(sundir.y, -0.07);
	float special_trick = 1.0 / (raydir.y * 1.0 + 0.1);
	float special_trick2 = 1.0 / (sundir.y * 11.0 + 1.0);
	float raysundt = pow(abs(dot(sundir, raydir)), 2.0);
	float sundt = pow(max(0.0, dot(sundir, raydir)), 8.0);
	float mymie = sundt * special_trick * 0.2;
	//float3 suncolor = mix(float3(1.0), max(float3(0.0), float3(1.0) - float3(5.5, 13.0, 22.4) / 22.4), special_trick2); // OLD
	float3 max_ = max(float3(0.0,0.0,0.0), float3(1.0,1.0,1.0) - float3(5.5, 13.0, 22.4) / 22.4);
	float3 suncolor = lerp(float3(1.0,1.0,1.0), max_, special_trick2);

	float3 bluesky= float3(5.5, 13.0, 22.4) / 22.4 * suncolor;
	float3 bluesky2 = max(float3(0.0,0.0,0.0), bluesky - float3(5.5, 13.0, 22.4) * 0.002 * (special_trick + -6.0 * sundir.y * sundir.y));
	bluesky2 *= special_trick * (0.24 + raysundt * 0.24);
	return bluesky2 * (1.0 + 1.0 * pow(1.0 - raydir.y, 3.0)) + mymie * suncolor;
} 
float3 getatm(float3 ray){
 	return extra_cheap_atmosphere(ray, normalize(float3(1.0,1.0,1.0))) * 0.5;
    
}

float sun(float3 ray){
 	float3 sd = normalize(float3(1.0,1.0,1.0));   
    return pow(max(0.0, dot(ray, sd)), 528.0) * 110.0;
}
float3 aces_tonemap(float3 color){	
	float3x3 m1 = float3x3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
	);
	float3x3 m2 = float3x3(
        1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602
	);
	//float3 v = m1 * color; // OLD
	float3 v = mul( color, m1 );
	float3 a = v * (v + 0.0245786) - 0.000090537;
	float3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
	//return pow(clamp(m2 * (a / b), 0.0, 1.0), float3(1.0 / 2.2));	// OLD
	//float3x3 zeroes = { float3( 0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0 ) };
	float3 value = mul( ( a / b ), m2 );
	float3 clamp_ = clamp( value, float3( 0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) );
	float3 div2 = float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2);
	return pow( clamp_, div2 );
}

float4 main(float4 position : SV_POSITION) : SV_TARGET { 
	// black
	//return float4( iTime % 1, 0, 0, 1 );
	float2 fragCoord = float2( position.xy );
	float4 fragColor;
	float2 uv = fragCoord.xy / iResolution.xy;
	uv.y = 1 - uv.y; // DirectX/HLSL rotate

	float waterdepth = 2.1;
	float3 wfloor = float3(0.0, -waterdepth, 0.0);
	float3 wceil = float3(0.0, 0.0, 0.0);
	float3 orig = float3(0.0, 2.0, 0.0);
	float3 ray = getRay(uv);
	float hihit = intersectPlane(orig, ray, wceil, float3(0.0, 1.0, 0.0));
    if ( ray.y >= -0.01 )
	{
        float3 C = getatm(ray) * 2.0 + sun(ray);
        //tonemapping
    	C = aces_tonemap(C);
     	fragColor = float4( C, 1.0 );   
        return fragColor;
    }
	float lohit = intersectPlane(orig, ray, wfloor, float3(0.0, 1.0, 0.0));
    float3 hipos = orig + ray * hihit;
    float3 lopos = orig + ray * lohit;
	float dist = raymarchwater(orig, hipos, lopos, waterdepth);
    float3 pos = orig + ray * dist;

	float3 N = normal(pos.xz, 0.001, waterdepth);
    float2 velocity = N.xz * (1.0 - N.y);
    N = lerp(float3(0.0, 1.0, 0.0), N, 1.0 / (dist * dist * 0.01 + 1.0));
	// Here artif couldbe, can check via return "float3(0, 0, 0)" or "N"
    float3 R = reflect(ray, N);
    float fresnel = (0.04 + (1.0-0.04)*(pow(1.0 - max(0.0, dot(-N, ray)), 5.0)));
	
    float3 C = fresnel * getatm(R) * 2.0 + fresnel * sun(R);
    //tonemapping
    C = aces_tonemap(C);
    
	fragColor = float4( C, 1.0 );
	return fragColor;
}
