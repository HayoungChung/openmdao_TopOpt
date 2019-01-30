from pylab import *

print("hello world")
if 0:
    for ii in range(1,7):
        BptsDat = loadtxt("cpp_" + str(ii) + ".txt") # coord_x coord_y, sens[0], delta_area, del_x*uku, sens[1]
        GptsDat = loadtxt("gpts_Sens_" + str(ii) + ".txt")# coord_x coord_y gpts_sens, area_elem 

        figure(0)
        clf()
        subplot(2,1,1)
        scatter(BptsDat[:,0], BptsDat[:,1], s=1, c=BptsDat[:,2]) # S_f
        axis("equal")
        colorbar()
        subplot(2,1,2)
        scatter(BptsDat[:,0], BptsDat[:,1], s=1, c=BptsDat[:,5]) # S_g
        axis("equal")
        colorbar()
        savefig("sensFigs/bpts_" + str(ii) + ".png")

        figure(1)
        clf()
        subplot(2,1,1)
        scatter(BptsDat[:,0], BptsDat[:,1], s=1, c=BptsDat[:,3]) # del_x
        axis("equal")
        colorbar()
        subplot(2,1,2)
        scatter(BptsDat[:,0], BptsDat[:,1], s=1, c=BptsDat[:,4]) # del_x*uku
        axis("equal")
        colorbar()
        savefig("areaFigs/bpts_" + str(ii) + ".png")
        


BptsDat0 = loadtxt("cpp_1.txt") # coord_x coord_y, sens[0], delta_area, del_x*uku, sens[1]
bpts_xy = loadtxt("bpts.txt")
sens_fg = loadtxt("Sens_fg.txt")

figure(0)
clf()
subplot(2,1,1)
scatter(BptsDat0[:,0],BptsDat0[:,1],s=5,c=BptsDat0[:,2])
colorbar()
subplot(2,1,2)
scatter(bpts_xy[:,0],bpts_xy[:,1],s=5,c=sens_fg[0])
colorbar()


figure(1)
clf()
subplot(2,1,1)
scatter(BptsDat0[:,0],BptsDat0[:,1],s=5,c=BptsDat0[:,5])
colorbar()
subplot(2,1,2)
scatter(bpts_xy[:,0],bpts_xy[:,1],s=5,c=sens_fg[1])
colorbar()

show()


print("bye world")