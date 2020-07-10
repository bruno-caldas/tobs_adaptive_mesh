"""
This module creates the adjoint object with ics as Fe
"""
# import dolfin as Fe
from dolfin import *
import BrunoDoc.properties as BProp
import BrunoDoc.forward_problem as FProb
from fenicstools import interpolate_nonmatching_mesh
from BrunoDoc.read_param_file import *
from BrunoDoc.filter_class import filter_obj

if linha[9][:2].upper() == 'DA': from dolfin_adjoint import *

delta = float(linha[2])
gap = float(linha[3])
radius = float(linha[4])
altura = 1.5

Reynolds = 300
v_average = Reynolds / (2*gap)
WF = 0.1
class AP(FProb.FP):
    minimize = True #False
    def __init__(self, hash=''):
        FProb.FP.__init__(self, hash)
        print("Creating the Adjoint Problem")

    def Funcional(self, rho, save_results=True): #, w, w2):
        mesh = rho.function_space().mesh()

        # self.filter_f = filter_obj(mesh, rmin=0.1, beta=1)
        # rho = self.filter_f.Rho_elem(rho)
        # rho.rename("ControlFiltered", "ControlFiltered")
        # self.file_filtrado << rho


        w = self.get_forward_solution(rho, save_results)
        (u, p) = split(w)

        # Absoluto COM KVV
        funcional1 = 1*(  inner(self.alpha(rho) * u, u) \
            +  0.5 * AP.mu *(
                inner(sym(grad(u)), sym(grad(u)))
                ) \
            # - inner(self.alpha(rho), (self.r_n-self.radius)**4) \
            )
        funcional1 *= dx

        return funcional1, w

    def get_adjoint_solution(self, rho, w):

        # w = self.get_forward_solution(rho)

        bc_hom = self.boundaries_cond()
        bc_hom[0].homogenize()

        adj = Function(self.W)
        adj_t = TrialFunction(self.W)
        (u_ad_t, p_ad_t) = split(adj_t)
        adj_tst = TestFunction(self.W)
        (v_ad, q_ad) = split(adj_tst)
        (u, p) = split(w)

        F_ad = (
                inner(div(u_ad_t), q_ad) \
                + inner(grad(u_ad_t), grad(v_ad)) \
                # - inner(grad(u_ad_t)*u, v_ad) \
                # + inner(grad(u).T* u_ad_t, v_ad) \
                + self.alpha(rho) * inner(u_ad_t, v_ad) \
                + inner(grad(p_ad_t), v_ad) \
                )*dx

        '''dJdu = derivative(self.funcional1, w) #FIXME
        '''
        dJdu = 1 *(
                2*self.alpha(rho) * inner(u, v_ad)* dx
                + 0.5*(
                    inner(sym(grad(u)), sym(grad(v_ad)))*dx
                    )
                )
        solve( F_ad == dJdu , adj, bc_hom)

        # rho = self.density_filter(rho)

        adj_u, adj_p = split(adj)
        adj_u_f = adj_u

        dmo = TestFunction( rho.function_space() )

        dJdm = (-1.*self.alphadash(rho)*inner(u,adj_u) + self.alphadash(rho)*inner(u,u))*dmo*dx
        # dJdm = (-1.*self.alphadash(rho)*inner(u,adj_u) )*dmo*dx
        adjfinal_1_resp = assemble(dJdm)

        return adjfinal_1_resp

