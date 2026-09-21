"""Cell-centred, conservative benchmark extension. Nondimensional c=a=rho=1.

Separate from the original endpoint-based Grid2D: all cells have volume h_x h_y.
Marshak incoming radiation is applied as a face flux, not a replaced PDE row.
"""
from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from trt_jfnk.models.opacity import PowerLawOpacity
from trt_jfnk.models.material import PolynomialMaterial
from trt_jfnk.discretization.diffusion import harmonic_mean


@dataclass
class FVProblem:
    case: str = 'marshak'
    nx: int = 32
    ny: int = 4
    cold: float = 0.2
    bath: float = 1.0
    sigma: float = 1.0
    exponent: float = 3.0
    contrast: float = 100.0
    floor: float = 0.01
    amplitude: float = 5.0
    pulse_end: float = 0.05

    def __post_init__(self):
        if self.case not in ('marshak', 'hotspot', 'inclusion'):
            raise ValueError('unknown case')
        if min(self.nx, self.ny) < 2 or min(self.cold, self.bath, self.contrast) <= 0:
            raise ValueError('positive temperatures/contrast and >=2 cells required')
        self.opacity = PowerLawOpacity(self.sigma, self.exponent, temperature_floor=self.floor)
        self.material = PolynomialMaterial(cv0=1.0, cv1=0.0)
        self.dx, self.dy = 1.0 / self.nx, 1.0 / self.ny
        self.size = self.nx * self.ny
        self.x = (jnp.arange(self.nx) + 0.5) * self.dx
        self.y = (jnp.arange(self.ny) + 0.5) * self.dy
        self.X, self.Y = jnp.meshgrid(self.x, self.y, indexing='ij')
        inclusion = (self.X - .62)**2 + (self.Y - .53)**2 < .14**2
        self.factor = jnp.where(inclusion, self.contrast, 1.0) if self.case == 'inclusion' else jnp.ones_like(self.X)

    def pack(self, E, T):
        return jnp.concatenate((E.ravel(), T.ravel()))

    def unpack(self, u):
        return u[:self.size].reshape(self.nx, self.ny), u[self.size:].reshape(self.nx, self.ny)

    def initial(self):
        T = jnp.full((self.nx, self.ny), self.cold)
        return self.pack(T**4, T)

    def coefficients(self, T):
        sigma = self.factor * self.opacity.absorption(T)
        return sigma, 1.0 / (3.0 * sigma)

    def face_fluxes(self, E, T):
        """Oriented physical fluxes F=-D grad E; reflecting outer faces except left drive."""
        _, D = self.coefficients(T)
        fx = jnp.zeros((self.nx + 1, self.ny), dtype=E.dtype)
        fy = jnp.zeros((self.nx, self.ny + 1), dtype=E.dtype)
        fx = fx.at[1:-1].set(-harmonic_mean(D[:-1], D[1:]) * (E[1:] - E[:-1]) / self.dx)
        fy = fy.at[:, 1:-1].set(-harmonic_mean(D[:, :-1], D[:, 1:]) * (E[:, 1:] - E[:, :-1]) / self.dy)
        if self.case == 'marshak':
            # E_b - 2D dE/dx = bath^4; eliminate E_b across a half cell.
            incoming = (self.bath**4 - E[0]) / (2.0 + 0.5 * self.dx / D[0])
            fx = fx.at[0].set(incoming)
        return fx, fy

    def source(self, time):
        if self.case == 'marshak':
            return jnp.zeros_like(self.X)
        q = self.amplitude * jnp.exp(-((self.X-.30)**2 + (self.Y-.43)**2)/(2*.09**2))
        return q * jnp.asarray(time <= self.pulse_end)

    def diffusion(self, E, T):
        fx, fy = self.face_fluxes(E, T)
        return -(fx[1:]-fx[:-1])/self.dx - (fy[:, 1:]-fy[:, :-1])/self.dy

    def residual(self, new, old, dt, time):
        E, T = self.unpack(new)
        E0, T0 = self.unpack(old)
        sigma, _ = self.coefficients(T)
        g = sigma * (E - T**4)
        return self.pack(E-E0-dt*(self.diffusion(E,T)-g+self.source(time)),
                         self.material.internal_energy(T)-self.material.internal_energy(T0)-dt*g)

    def admissible(self, u):
        E, T = self.unpack(u)
        return jnp.all(jnp.isfinite(u)) & jnp.all(E >= 0) & jnp.all(T > 1.e-8)

    def energy(self, u):
        E, T = self.unpack(u)
        return jnp.sum(E+self.material.internal_energy(T))*self.dx*self.dy

    def input_power(self, u, time):
        E, T = self.unpack(u)
        fx, fy = self.face_fluxes(E,T)
        return ((jnp.sum(fx[0])-jnp.sum(fx[-1]))*self.dy
                +(jnp.sum(fy[:,0])-jnp.sum(fy[:,-1]))*self.dx
                +jnp.sum(self.source(time))*self.dx*self.dy)

    def local_blocks(self, u, dt):
        E,T = self.unpack(u)
        sigma,_ = self.coefficients(T)
        sp = self.factor*self.opacity.absorption_derivative(T)
        gt = sp*(E-T**4)-4*sigma*T**3
        return tuple(v.ravel() for v in (1+dt*sigma, dt*gt, -dt*sigma,
                                        self.material.heat_capacity(T)-dt*gt))

    def negative_diffusion_matrix(self, u):
        """Sparse -L with frozen coefficients; affine boundary drive excluded."""
        _,T = self.unpack(u)
        _,D = self.coefficients(T)
        D=np.asarray(D); idx=np.arange(self.size).reshape(self.nx,self.ny)
        rows=[]; cols=[]; vals=[]
        for axis,h in ((0,self.dx),(1,self.dy)):
            a,b=(idx[:-1,:],idx[1:,:]) if axis==0 else (idx[:,:-1],idx[:,1:])
            da,db=(D[:-1,:],D[1:,:]) if axis==0 else (D[:,:-1],D[:,1:])
            w=(2*da*db/(da+db)/h**2).ravel(); a=a.ravel(); b=b.ravel()
            for rr,cc,vv in ((a,a,w),(b,b,w),(a,b,-w),(b,a,-w)):
                rows.extend(rr); cols.extend(cc); vals.extend(vv)
        if self.case=='marshak':
            w=1/(2+0.5*self.dx/D[0])/self.dx
            rows.extend(idx[0]); cols.extend(idx[0]); vals.extend(w)
        return coo_matrix((vals,(rows,cols)),shape=(self.size,self.size)).tocsc()

    def preconditioner(self, dt, kind):
        if kind=='none':
            return None
        def factory(u, context):
            a,b,c,d=self.local_blocks(u,dt)
            if kind=='local-block':
                determinant=a*d-b*c
                if bool(jnp.any(jnp.abs(determinant)<1e-20)):
                    raise RuntimeError('singular local preconditioner')
                def physical(v):
                    x,y=v[:self.size],v[self.size:]
                    return jnp.concatenate(((d*x-b*y)/determinant,(-c*x+a*y)/determinant))
            elif kind=='schur':
                # CPU sparse-direct Schur reference, not AMG and not GPU-resident.
                a,b,c,d=map(np.asarray,(a,b,c,d))
                if np.any(np.abs(d)<1e-20):
                    raise RuntimeError('singular material block')
                factor=splu(diags(a-b*c/d)+dt*self.negative_diffusion_matrix(u))
                def physical(v):
                    v=np.asarray(v); x,y=v[:self.size],v[self.size:]
                    e=factor.solve(x-b*y/d)
                    return jnp.asarray(np.concatenate((e,(y-c*e)/d)))
            else:
                raise ValueError('unknown preconditioner')
            return lambda v: physical(context.residual_scale*v)/context.state_scale
        return factory
