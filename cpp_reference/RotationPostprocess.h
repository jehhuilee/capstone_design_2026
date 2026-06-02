// RotationPostprocess.h
// ─────────────────────────────────────────────────────────────────────────────
// 손목/팔뚝 과도 트위스트(Over-twisting) + 플래싱(Flashing) 제거 회전 후처리.
// Python(rotation_postprocess.py)에서 실측 검증된 최적 파이프라인의 C++ 이식 참고본.
//   (1) Quaternion Continuity  (2) Savitzky-Golay smoothing
//   (3) Swing-Twist ROM (unwrap)  (4) Angular-velocity clamp
// 의존성 없음(헤더 온리). 좌표/쿼터니언 규약은 [w,x,y,z], 우수계 가정.
// Unreal 이식 시 Quat ↔ FQuat(주의: FQuat은 [x,y,z,w]) 매핑만 교체하면 됨.
// ─────────────────────────────────────────────────────────────────────────────
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846   // MSVC 등에서 미정의 대비
#endif

namespace rpp {

struct Quat {            // w + xi + yj + zk
    double w=1, x=0, y=0, z=0;
    Quat() {}
    Quat(double w_,double x_,double y_,double z_):w(w_),x(x_),y(y_),z(z_){}
};
struct Vec3 { double x=0,y=0,z=0; };

inline double dot(const Quat&a,const Quat&b){return a.w*b.w+a.x*b.x+a.y*b.y+a.z*b.z;}
inline Quat   neg(const Quat&a){return {-a.w,-a.x,-a.y,-a.z};}
inline Quat   conj(const Quat&a){return {a.w,-a.x,-a.y,-a.z};}
inline double norm(const Quat&a){return std::sqrt(dot(a,a));}
inline Quat   normalize(const Quat&a){double n=norm(a); return n<1e-12?Quat():Quat{a.w/n,a.x/n,a.y/n,a.z/n};}
inline Quat   mul(const Quat&a,const Quat&b){
    return { a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z,
             a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
             a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
             a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w };
}

// rotvec(axis-angle, 길이=각도[rad]) ↔ Quat -------------------------------------
inline Quat rotvecToQuat(const Vec3&rv){
    double a=std::sqrt(rv.x*rv.x+rv.y*rv.y+rv.z*rv.z);
    if(a<1e-12) return Quat();
    double s=std::sin(a*0.5)/a;
    return { std::cos(a*0.5), rv.x*s, rv.y*s, rv.z*s };
}
inline Vec3 quatToRotvec(const Quat&q_){
    Quat q = q_.w<0 ? neg(q_) : q_;          // 0..pi 범위
    double vn=std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z);
    if(vn<1e-12) return {0,0,0};
    double ang=2.0*std::atan2(vn,q.w);
    double k=ang/vn;
    return { q.x*k, q.y*k, q.z*k };
}

inline Quat slerp(const Quat&a,Quat b,double t){
    double d=dot(a,b);
    if(d<0){b=neg(b);d=-d;}
    if(d>0.9995){ Quat r{a.w+t*(b.w-a.w),a.x+t*(b.x-a.x),a.y+t*(b.y-a.y),a.z+t*(b.z-a.z)}; return normalize(r);}
    double th0=std::acos(std::clamp(d,-1.0,1.0)), s0=std::sin(th0);
    double c0=std::sin((1-t)*th0)/s0, c1=std::sin(t*th0)/s0;
    return { c0*a.w+c1*b.w, c0*a.x+c1*b.x, c0*a.y+c1*b.y, c0*a.z+c1*b.z };
}

// (1) 쿼터니언 연속성 ----------------------------------------------------------
//   q_t·q_{t-1} < 0 이면 q_t → -q_t. 표현상 180°/360° 점프 제거.
inline void enforceContinuity(std::vector<Quat>& q){
    for(size_t t=1;t<q.size();++t)
        if(dot(q[t],q[t-1])<0) q[t]=neg(q[t]);
}

// (5/winner) Savitzky-Golay: 윈도우 다항 최소제곱 평활 --------------------------
//   계수는 Vandermonde 정규방정식 (AᵀA)c = Aᵀe0 로 1회 산출(가장자리는 반사 패딩).
inline std::vector<double> sgCoeffs(int half,int poly){
    int W=2*half+1, P=poly+1;
    std::vector<std::vector<double>> A(W,std::vector<double>(P));
    for(int i=0;i<W;++i){double xi=i-half,p=1; for(int j=0;j<P;++j){A[i][j]=p;p*=xi;}}
    std::vector<std::vector<double>> N(P,std::vector<double>(P,0)); // AᵀA
    for(int a=0;a<P;++a)for(int b=0;b<P;++b)for(int i=0;i<W;++i)N[a][b]+=A[i][a]*A[i][b];
    // (AᵀA)^{-1} 의 0행만 필요 → e0 풀이 (Gauss elimination)
    std::vector<double> rhs(P,0); rhs[0]=1;
    for(int c=0;c<P;++c){
        int piv=c; for(int r=c+1;r<P;++r) if(std::fabs(N[r][c])>std::fabs(N[piv][c])) piv=r;
        std::swap(N[c],N[piv]); std::swap(rhs[c],rhs[piv]);
        double dv=N[c][c];
        for(int j=0;j<P;++j)N[c][j]/=dv; rhs[c]/=dv;
        for(int r=0;r<P;++r) if(r!=c){double f=N[r][c]; for(int j=0;j<P;++j)N[r][j]-=f*N[c][j]; rhs[r]-=f*rhs[c];}
    }
    std::vector<double> coeff(W,0);                       // h_i = Σ_j (N^{-1})_{0j} x_i^j
    for(int i=0;i<W;++i){double xi=i-half,p=1; for(int j=0;j<P;++j){coeff[i]+=rhs[j]*p;p*=xi;}}
    return coeff;
}
inline std::vector<double> sgSmooth1D(const std::vector<double>& s,int half,int poly){
    int n=(int)s.size(); if(n< poly+2 || half<1) return s;
    auto h=sgCoeffs(half,poly); std::vector<double> o(n);
    for(int t=0;t<n;++t){double acc=0;
        for(int k=-half;k<=half;++k){int idx=t+k; if(idx<0)idx=-idx; if(idx>=n)idx=2*n-2-idx; // 반사
            acc+=h[k+half]*s[std::clamp(idx,0,n-1)];}
        o[t]=acc;}
    return o;
}
//   연속화한 쿼터니언 4성분 평활 후 재정규화.
inline void savgolQuat(std::vector<Quat>& q,int window,int poly){
    int n=(int)q.size(); int half=window/2; if(n<poly+2||half<1) return;
    std::vector<double> cw(n),cx(n),cy(n),cz(n);
    for(int i=0;i<n;++i){cw[i]=q[i].w;cx[i]=q[i].x;cy[i]=q[i].y;cz[i]=q[i].z;}
    cw=sgSmooth1D(cw,half,poly);cx=sgSmooth1D(cx,half,poly);
    cy=sgSmooth1D(cy,half,poly);cz=sgSmooth1D(cz,half,poly);
    for(int i=0;i<n;++i) q[i]=normalize(Quat{cw[i],cx[i],cy[i],cz[i]});
}

// (2/winner) Swing-Twist ROM (unwrap) ------------------------------------------
//   q = swing * twist (twist는 axis 둘레). twist각 φ를 시간축 unwrap → soft-clamp → 재조립.
inline void swingTwistDecompose(const Quat&q,const Vec3&axis,Quat&swing,double&phi){
    double d=q.x*axis.x+q.y*axis.y+q.z*axis.z;            // v·axis
    Quat twist=normalize(Quat{q.w, d*axis.x, d*axis.y, d*axis.z});
    phi=2.0*std::atan2(d,q.w);
    phi=std::fmod(phi+M_PI,2*M_PI); if(phi<0)phi+=2*M_PI; phi-=M_PI;   // (-π,π]
    swing=mul(q,conj(twist));
}
inline void twistROM(std::vector<Quat>& q,const Vec3&axis,double twMinRad,double twMaxRad,
                     int smoothWindow,int poly=3){
    int n=(int)q.size(); if(n==0) return;
    Vec3 a=axis; double an=std::sqrt(a.x*a.x+a.y*a.y+a.z*a.z); a={a.x/an,a.y/an,a.z/an};
    std::vector<Quat> swing(n); std::vector<double> phi(n);
    for(int t=0;t<n;++t) swingTwistDecompose(q[t],a,swing[t],phi[t]);
    for(int t=1;t<n;++t){                                  // unwrap
        double d=phi[t]-phi[t-1];
        while(d> M_PI){phi[t]-=2*M_PI; d=phi[t]-phi[t-1];}
        while(d<-M_PI){phi[t]+=2*M_PI; d=phi[t]-phi[t-1];}
    }
    if(smoothWindow>=5) phi=sgSmooth1D(phi,smoothWindow/2,poly);
    for(int t=0;t<n;++t){
        double p=std::clamp(phi[t],twMinRad,twMaxRad);
        double h=p*0.5; Quat tw{std::cos(h),std::sin(h)*a.x,std::sin(h)*a.y,std::sin(h)*a.z};
        q[t]=mul(swing[t],tw);
    }
}

// (3/winner) 각속도 클램핑 ------------------------------------------------------
inline void angVelClamp(std::vector<Quat>& q,double maxRadPerFrame){
    for(size_t t=1;t<q.size();++t){
        double d=std::clamp(std::fabs(dot(q[t-1],q[t])),-1.0,1.0);
        double ang=2.0*std::acos(d);
        if(ang>maxRadPerFrame && ang>1e-9) q[t]=slerp(q[t-1],q[t],maxRadPerFrame/ang);
    }
}

// ── 통합: 단일 관절 시퀀스 후처리 (검증된 승자 파이프라인) ────────────────────
struct StabilizeParams {
    int    sgWindow        = 9;           // 홀수
    int    sgPoly          = 3;
    bool   applyTwistROM   = false;       // 트위스트 제한 관절(어깨/팔꿈치/손목)만 true
    Vec3   twistAxis       = {1,0,0};     // 본 장축(미지정 시 데이터 PCA 주축 권장)
    double twistMinDeg     = -90, twistMaxDeg = 90;
    double maxDegPerFrame  = 30;          // 0 이면 각속도 클램프 생략
};
inline void stabilizeJoint(std::vector<Quat>& q,const StabilizeParams& p){
    enforceContinuity(q);                                 // (1)
    savgolQuat(q,p.sgWindow,p.sgPoly);                    // (5)
    if(p.applyTwistROM)                                   // (2)
        twistROM(q,p.twistAxis,p.twistMinDeg*M_PI/180.0,p.twistMaxDeg*M_PI/180.0,p.sgWindow|1,p.sgPoly);
    if(p.maxDegPerFrame>0)                                // (3)
        angVelClamp(q,p.maxDegPerFrame*M_PI/180.0);
}

} // namespace rpp
