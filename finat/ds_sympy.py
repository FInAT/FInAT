import sympy
import numpy
import FIAT

u = FIAT.ufc_cell("quadrilateral")
ct = u.topology

# symbols x00, x01, x11, etc for physical vertex coords
vs = numpy.asarray(list(zip(sympy.symbols('x:4'), sympy.symbols('y:4'))))
xx = numpy.asarray(sympy.symbols("x,y"))

#phys_verts = numpy.array(u.vertices)
phys_verts = numpy.array([[0.0, 0.0], [0.0, 1.0], [2.0, 0.0], [1.5, 3.0]])
sdict = {vs[i, j]: phys_verts[i, j] for i in range(4) for j in range(2)}

ts = numpy.zeros((4, 2), dtype=object)
for e in range(4):
    v0id, v1id = ct[1][e][:]
    for j in range(2):
        ts[e, :] = vs[v1id, :] - vs[v0id, :]


ns = numpy.zeros((4, 2), dtype=object)
for e in (0, 3):
    ns[e, 0] = -ts[e, 1]
    ns[e, 1] = ts[e, 0]

for e in (1, 2):
    ns[e, 0] = ts[e, 1]
    ns[e, 1] = -ts[e, 0]


def xysub(x, y):
    return {x[0]: y[0], x[1]: y[1]}
    
# midpoints of each edge
xstars = numpy.zeros((4, 2), dtype=object)
for e in range(4):
    v0id, v1id = ct[1][e][:]
    xstars[e, :] = (vs[v0id, :] + vs[v1id])/2

lams = [(xx-xstars[i, :]) @ ns[i, :] for i in range(4)]

RV = (lams[0] - lams[1]) / (lams[0] + lams[1])
RH = (lams[2] - lams[3]) / (lams[2] + lams[3])
Rs = [RV, RH]

xis = []
sgn = -1
for e in range(4):
    dct = xysub(xx, xstars[e, :])
    i = 2*((3-e)//2)
    j = i + 1
    xi = lams[i] * lams[j] * (1+ (-1)**(e+1) * Rs[e//2]) / lams[i].subs(dct) / lams[j].subs(dct) / 2
    xis.append(xi)

  
#sympy.plotting.plot3d(xis[3].subs(sdict), (xx[0], 0, 1), (xx[1], 0, 1))

# These check out!
# for xi in xis:
#     for e in range(4):
#         print(xi.subs(xysub(xx, phys_verts[e, :])).subs(sdict))
#     for e in range(4):
#         print(xi.subs(xysub(xx, xstars[e, :])).subs(sdict))
#     print()


d = xysub(xx, vs[0, :])

r = lams[1] * lams[3] / lams[1].subs(d) / lams[3].subs(d)

d = xysub(xx, vs[2, :])

r -= lams[0] * lams[3] / lams[0].subs(d) / lams[3].subs(d)

d = xysub(xx, vs[3, :])

r += lams[0] * lams[2] / lams[0].subs(d) / lams[2].subs(d)

d = xysub(xx, vs[1, :])

r -= lams[1] * lams[2] / lams[1].subs(d) / lams[2].subs(d)

R = r - sum([r.subs(xysub(xx, xstars[i, :])) * xis[i] for i in range(4)])


# for e in range(4):
#     print(R.subs(sdict).subs({xx[0]: phys_verts[e, 0], xx[1]: phys_verts[e, 1]}))
# for e in range(4):
#     print(R.subs({xx[0]: xstars[e, 0], xx[1]: xstars[e, 1]}).subs(sdict))

    
n03 = numpy.array([[0, -1], [1, 0]]) @ (vs[3, :] - vs[0, :])
lam03 = (xx - vs[0, :]) @ n03

n12 = numpy.array([[0, -1], [1, 0]]) @ (vs[2, :] - vs[1, :])
lam12 = (xx - vs[2, :]) @ n12


def ds1():
    phi0tilde = lam12 - lam12.subs({xx[0]: vs[3, 0], xx[1]: vs[3, 1]}) * (1 + R) / 2
    phi1tilde = lam03 - lam03.subs({xx[0]: vs[2, 0], xx[1]: vs[2, 1]}) * (1 - R) / 2
    phi2tilde = lam03 - lam03.subs({xx[0]: vs[1, 0], xx[1]: vs[1, 1]}) * (1 - R) / 2
    phi3tilde = lam12 - lam12.subs({xx[0]: vs[0, 0], xx[1]: vs[0, 1]}) * (1 + R) / 2

    phis = []
    for i, phitilde in enumerate([phi0tilde, phi1tilde, phi2tilde, phi3tilde]):
        phi = phitilde / phitilde.subs({xx[0]: vs[i, 0], xx[1]: vs[i, 1]})
        phis.append(phi)

    phis = numpy.asarray(phis)
    return phis

def ds2_sympy():
    xx2xstars = [xysub(xx, xstars[i]) for i in range(4)]
    v_phitildes = [(lams[1] * lams[3]
                    - lams[3].subs(xx2xstars[2])
                    / lams[0].subs(xx2xstars[2])
                    * lams[0] * lams[1] * (1-RH) / 2
                    - lams[1].subs(xx2xstars[0])
                    / lams[2].subs(xx2xstars[0])
                    * lams[2] * lams[3] * (1-RV) / 2),
                   (lams[1] * lams[2]
                    - lams[2].subs(xx2xstars[3])
                    / lams[0].subs(xx2xstars[3])
                    * lams[0] * lams[1] * (1+RH) / 2
                    - lams[1].subs(xx2xstars[0])
                    / lams[3].subs(xx2xstars[0])
                    * lams[2] * lams[3] * (1-RV) / 2),
                   (lams[0] * lams[3]
                    - lams[3].subs(xx2xstars[2])
                    / lams[1].subs(xx2xstars[2])
                    * lams[0] * lams[1] * (1-RH) / 2
                    - lams[0].subs(xx2xstars[1])
                    / lams[2].subs(xx2xstars[1])
                    * lams[2] * lams[3] * (1+RV) / 2),
                   (lams[0] * lams[2]
                    - lams[2].subs(xx2xstars[3])
                    / lams[1].subs(xx2xstars[3])
                    * lams[0] * lams[1] * (1+RH) / 2
                    - lams[0].subs(xx2xstars[1])
                    / lams[3].subs(xx2xstars[1])
                    * lams[2] * lams[3] * (1+RV) / 2)]
    phis_v = [phitilde_v / phitilde_v.subs(xysub(xx, vs[i, :]))
              for i, phitilde_v in enumerate(v_phitildes)]
    
    e_phitildes = [lams[2] * lams[3] * (1-RV) / 2,
                   lams[2] * lams[3] * (1+RV) / 2,
                   lams[0] * lams[1] * (1-RH) / 2,
                   lams[0] * lams[1] * (1+RH) / 2]

    phis_e = [ephi / ephi.subs(xx2xstars[i]) for i, ephi in enumerate(e_phitildes)]

    return numpy.asarray(phis_v + phis_e)


for phi in ds2_sympy():
    for e in range(4):
        print(phi.subs(sdict).subs({xx[0]: phys_verts[e, 0], xx[1]: phys_verts[e, 1]}))
    for e in range(4):
        print(phi.subs({xx[0]: xstars[e, 0], xx[1]: xstars[e, 1]}).subs(sdict))
    print()
    
