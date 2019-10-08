# stress

from pylab import *
from openmdao.api import ExplicitComponent
sys.path.append('../Density_OpenLSTO')

from pyBind import py_FEA

class VMStressComp(ExplicitComponent): # ERROR ~ 1%
    def initialize(self):
        self.options.declare('fea_solver', types=py_FEA)
        self.options.declare('nelx', types=int, )
        self.options.declare('nely', types=int, )
        self.options.declare('length_x', types=float, )
        self.options.declare('length_y', types=float, )
        self.options.declare('order', types=float)
        self.options.declare('E', types=float)
        self.options.declare('nu', types=float)
        
    def setup(self):
        fea_solver = self.fea_solver = self.options['fea_solver']
        nelx = self.nelx = self.options['nelx']
        nely = self.nely = self.options['nely']
        length_x = self.length_x = self.options['length_x']
        length_y = self.length_y = self.options['length_y']
        E = self.E = self.options['E']
        nu = self.nu = self.options['nu']

        order = self.order = self.options['order'] # order of quadrature
        self.nELEM = nELEM = nelx * nely
        self.nNODE = nNODE = (nelx+1) * (nely+1)
        self.nDOF = nDOF = 2 * nNODE
        self.gpe  = order ** 2

        self.add_input('disp', shape=nDOF)
        self.add_input('density', shape=nELEM)
        self.add_output('vmStress', shape=nELEM)

        self.elem_id = elem_id = fea_solver.get_mesh()[2]
        rows = np.matlib.repmat(arange(nELEM), 8, 1)
        rows = rows.flatten(order='F')
        cols = elem_id.flatten()
        
        self.declare_partials('vmStress','disp', rows = rows, cols = cols)

        rows = arange(nELEM)
        self.declare_partials('vmStress','density', rows = rows, cols = rows)
        self.B0 = self._Bmatrix_centroid()

    def compute(self,inputs,outputs):
        u = array(inputs['disp'])
        rho = array(inputs['density'])

        vm = self._vonMises(u, rho)[0]
        outputs['vmStress'] = vm

    def compute_partials(self, inputs, partials):
        u = inputs['disp']
        rho = inputs['density']
        B0 = self.B0 
        Cijkl = zeros((3,3))
        Cijkl = array([[1., self.nu, 0.], [self.nu, 1., 0.], [0., 0., 0.5*(1.-self.nu)]])
        Cijkl *= self.E/(1.-self.nu*self.nu)
        CB0 = Cijkl.dot(B0)        

        vm, dev = self._vonMises(u, rho)
        J2 = np.power(vm,2.)/3.

        # for partial vm / partial disp
        val = zeros(self.nELEM * 8)
        for ee in range(self.nELEM):
            if ((rho[ee] < 0.01) or (abs(J2[ee]) < 1e-7)):
                continue
            for qqq in range(8):
                val[8*ee + qqq] = dev[ee,0]*CB0[0,qqq] + dev[ee,1]*CB0[1,qqq] + 2*dev[ee,2]*CB0[2,qqq] - 1./3.*(dev[ee,0]+dev[ee,1])*(CB0[0,qqq]+CB0[1,qqq])
                val[8*ee + qqq] *= sqrt(3.)/2. / sqrt(J2[ee]) * rho[ee]
                
        partials['vmStress','disp'] = val
        
        # for partial vm / partial rho
        val = zeros(self.nELEM)
        for ee in range(self.nELEM):
            eid = self.elem_id[ee]
            uel = u[eid]
            if (rho[ee] < 0.01): #(abs(J2[ee]) < 1e-7):
                continue
            CBU = CB0.dot(uel)
            val[ee] = vm[ee]/rho[ee]

        partials['vmStress','density'] = val

    def _vonMises(self, u, rho):
        self.fea_solver.set_stress(u)
        sigma = zeros((self.nELEM,3))
        for ee in range(self.nELEM):
            (loc_, sig_) = self.fea_solver.get_stress(ee)
            sig_ = array(sig_)
            if (rho[ee] < 0.01):
                sigma[ee] = sig_ * 0.0
            else:
                sigma[ee] = sig_ * rho[ee]

        dev = sigma  # deviatoric stress
        dev[:,0] -= 1./3.*(sigma[:,0]+sigma[:,1])
        dev[:,1] -= 1./3.*(sigma[:,0]+sigma[:,1])
        J2 = multiply(dev[:,0],dev[:,0]) + multiply(dev[:,1],dev[:,1]) + 2*multiply(dev[:,2],dev[:,2]) 
        J2 *= 0.5
        return (sqrt(3.*J2), dev)

    def _Bmatrix_centroid(self):
        # ASSUMPTION: 
        # superconvergent point = centroid of Q4
        # structured mesh

        Bmatrix = zeros([3,8])
        lx_elx = self.length_x/float(self.nelx)
        ly_ely = self.length_y/float(self.nely)

        dN_dx = 2./lx_elx * 0.25 * array([-1., 1., 1., -1.])
        dN_dy = 2./ly_ely * 0.25 * array([-1., -1., 1., 1.])        

        Bmatrix[0,0::2] = dN_dx
        Bmatrix[1,1::2] = dN_dy
        Bmatrix[2,0::2] = dN_dy
        Bmatrix[2,1::2] = dN_dx

        return Bmatrix


class pVmComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('pval', types=float)
        self.options.declare('nelx', types=int)
        self.options.declare('nely', types=int)
        
    def setup(self):
        pval = self.pval = self.options['pval']
        nelx = self.nelx = self.options['nelx']
        nely = self.nely = self.options['nely']
        nELEM = self.nELEM = nelx*nely

        self.add_input('x', shape=nELEM)
        self.add_output('xp', shape=nELEM)
        ran1 = arange(nELEM)
        
        self.declare_partials(of='xp', wrt='x', rows=ran1, cols=ran1)
    
    def compute(self, inputs, outputs):
        vm = inputs['x']
        vmp = np.power(vm, self.pval)
        outputs['xp'] = vmp
        
    
    def compute_partials(self, inputs, partials):
        vm = inputs['x']
        val = np.power(vm, self.pval-1.) * self.pval 
        partials['xp','x'] = val
            

class pnormComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('pval', types=float)
        self.options.declare('nelx', types=int)
        self.options.declare('nely', types=int)
        
    def setup(self):
        pval = self.pval = self.options['pval']
        nelx = self.nelx = self.options['nelx']
        nely = self.nely = self.options['nely']
        nELEM = self.nELEM = nelx*nely

        self.add_input('x', shape=1)
        self.add_output('pnorm', shape=1)
        
        self.declare_partials(of='pnorm', wrt='x', rows=[0], cols=[0])
    
    def compute(self, inputs, outputs):
        x = inputs['x']
        pnorm = np.power(x, 1./self.pval)
        outputs['pnorm'] = pnorm        
    
    def compute_partials(self, inputs, partials):
        x = inputs['x']
        val = np.power(x, 1./self.pval-1.) * 1./self.pval 
        partials['pnorm','x'] = val

class BodyIntegComp(ExplicitComponent):
    # assume a constructed mesh
    # NOTE: integration point = centroid
    def initialize(self):
        self.options.declare('nelx', types=int)
        self.options.declare('nely', types=int)
        self.options.declare('length_x', types=float)
        self.options.declare('length_y', types=float)

    def setup(self):
        nelx = self.nelx = self.options['nelx']
        nely = self.nely = self.options['nely']
        length_x = self.length_x = self.options['length_x']
        length_y = self.length_y = self.options['length_y']
        nELEM = self.nELEM = nelx * nely
        lx_elx = self.lx_elx = length_x/float(nelx)
        ly_ely = self.ly_ely = length_y/float(nely)

        detJ = self.detJ = lx_elx * ly_ely / 4.0
        self.add_input('x', shape=nELEM)
        self.add_output('y', shape=1)

        val = ones(nELEM)*4.0*detJ
        self.declare_partials(of='y', wrt='x', rows=zeros(nELEM), cols=arange(nELEM), val=val)

    def compute(self, inputs, outputs):
        x = inputs['x']
        int_x = 0.
        for ee in range(self.nELEM):
            int_x += x[ee]*4.0*self.detJ
        outputs['y'] = int_x

    def compute_partials(self, inputs, partials):
        pass