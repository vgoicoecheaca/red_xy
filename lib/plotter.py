import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib as mpl 
import cmocean
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic_2d
import numpy as np

plt.style.use('/Users/victor/mystyle.mlstyle')

class Plotter():
    sipm_xs = np.linspace(-2.5+(1.25/2),2.5-(1.25/2),4)             # center of pdm positions
    sipm_ys = np.linspace(-2.5+(0.833333/2),2.5-(0.833333/2),6)
    sipm_w , sipm_h = sipm_xs[1] - sipm_xs[0], sipm_ys[1]-sipm_ys[0]
 
    def __init__(self,manager):
        self.m       = manager
        self.source  = self.m.config('run','source','str')

        if self.source == 'Am':
            self.rs1 = (0,1000)
            self.rs2 = (0,20000)
        if self.source == 'bg':
            self.rs1 = (0,1200)
            self.rs2 = (0,2500)
        if self.source == 'Na':
            self.rs1 = (0,2000)
            self.rs2 = (0,40000)
        self.rt      = (0,80) 
        self.rs1s2   = (self.rs1,self.rs2)
        self.rover   = (self.rs1,(0,50))
        self.rf90    = (self.rs1,(0,1.1))
        self.rts2s1  = (self.rt,(0,27))
        self.rts1    = (self.rt,self.rs1)
        self.rts2    = (self.rt,self.rs2)
        self.max_x   = self.m.config('run','max_x','float')
        self.max_y   = self.m.config('run','max_y','float')  
        self.lifetime = self.m.config('run','lifetime','int')*60
        self.lifetime_bg = self.m.config('run','lifetime_bg','int')*60
    
    def inspect(self,s1,s2,tdrift,tdrift_s2,pre_pars = [[],[]]):
        print("Inspecting and plotting for initial cuts...")
        tau,ps1,ps2 = 1,[],[]
        if self.m.config('plot','s1','bool'):
            ps1 = self.hist1d(s1, self.rs1, ["p","S1 [PE]", "Events"] ,bins = 30,fit = True)
        if self.m.config('plot','s2','bool'): 
            ps2 = self.hist1d(s2, self.rs2, ["p","S2 [PE]", "Events"], bins = 30,fit = True) 
        if self.m.config('plot','t','bool'): 
            self.hist1d(tdrift, self.rt, [" ", r"$t_d$ [us]", "Events"]) # tdrift
        if self.m.config('plot','s1s2','bool'):
            if len(pre_pars[0])!= 0:
                ps1,ps2 = pre_pars[0],pre_pars[1]
            self.hist2d(s1,s2, self.rs1s2,["Events","S1 [PE]", "S2 [PE]"],False,'sigmas', pars = [ps1,ps2]) # s1 vs s2 
        if self.m.config('plot','ts1s2','bool'):
            tau = self.hist2d(tdrift_s2,s2/s1, self.rts2s1,["Events", r"$t_d$ [us]", "S2/S1"], True, 'exp', pars = None)
        if self.m.config('plot','ts1','bool'):
            self.hist2d(tdrift_s2, s1, self.rts1, ["Events", r"$t_d$ [us]", "S1 [PE]"]) # tdrift vs s1
        if self.m.config('plot','ts2','bool'):
            self.hist2d(tdrift_s2, s2, self.rts2, ["Events", r"$t_d$ [us]", "S2 [PE]","Fit"], False , None,None) # tdrift vs s2
            self.hist2d(tdrift_s2, s2/np.exp(-tdrift_s2/tau), self.rts2, ["Events", r"$t_d$ [us]", "S2 [PE]","Fit"], False, 'line',None) # tdrift vs s2
        
        return tau,ps1,ps2 

    def hist1d(self,var, range, labels,bins=30, norm=None, fit = False):
        if norm!=None:
            print("Norm is not none")
            n,bins = np.histogram(var,bins =bins,range=range)
            bin = bins[:-1] + np.diff(bins)/2
            plt.step(bin,np.divide(n,norm),where="mid",label =labels[0])
        else:  
            n, bins, _ = plt.hist(var, bins = bins, range=range, histtype = 'step',label = labels[0])
        plt.xlabel(labels[1])
        plt.ylabel(labels[2])
        if fit:
            xs, gauss, pars = self.m.helper.gauss_fit(n,bins)
            plt.plot(xs, gauss, "r-", label = r"Gauss fit $\pm 2 \sigma$")
        plt.legend(loc = 'best')
        plt.grid(True, axis = 'both')
        plt.show()
        return pars if fit else 1
    
    def coord_reco(self,reco,mc,bins,range,labels,unc,unc_mc,bg=[]): 
        # reconstruciton 
        h_cnn,e_cnn = np.histogram(reco,bins=bins,range=range)
        yerr_cnn = np.divide(unc,np.linspace(range[0],range[-1],bins))*np.divide(h_cnn,self.lifetime)
        hits = h_cnn / self.lifetime
        yerr = yerr_cnn
        bin = e_cnn[:-1] + np.diff(e_cnn)/2
        # if Bg subtraction  
        if len(bg)!=0:
            h_bg, _ = np.histogram(bg,bins=bins,range=range) 
            yerr_cnn_bg = np.divide(unc,np.linspace(range[0],range[-1],bins))*(h_bg/self.lifetime_bg)
            hits = hits - (h_bg/self.lifetime_bg)
            yerr  = np.sqrt((np.square(yerr_cnn_bg) + np.square(yerr_cnn))) 
        plt.errorbar(bin,hits,yerr = yerr,fmt='ok',label = labels[0])
      
        #mc,uncertainty due to place holder, pm 0.5cm, and convoluted with the model's resolution at corresponding position 
        h_mc,e_mc  = np.histogram(mc,bins=bins,range=range)
        mc_smears = self.m.helper.conv(h_mc,e_mc)
        h_mc,e_mc  = np.histogram(mc_smears,bins=bins,range=range)
        lifetime_mc = (len(mc_smears)*self.lifetime)/len(reco)
        yerr_mc = unc_mc*np.divide(h_mc,lifetime_mc)
        h_mc    = np.divide(h_mc,lifetime_mc)
        plt.bar(bin, h_mc,width=np.diff(e_mc),color = 'deepskyblue',alpha=0.7,label='MC') 
        y_low = np.append(h_mc-yerr_mc,(h_mc-yerr_mc)[-1])
        y_high = np.append(h_mc+yerr_mc,(h_mc+yerr_mc)[-1])
        plt.fill_between(e_mc,y_low,y_high,facecolor= "none",label= 'Uncertainty',step='post',hatch= '//') 
    
        plt.xlabel(labels[1])
        plt.ylabel(labels[2])
        plt.legend(loc = 'upper left')
        plt.grid(True, axis = 'both')
        plt.xlim(range[0],range[-1])
        plt.ylim(bottom=0)
        plt.savefig('save/coord'+labels[1][0]+'.png',dpi=400)
        plt.show()
        #fig.add_axes([0.1,0.1,0.8,0.2])
        #plt.fill_between(bin,-yerr_mc,yerr_mc,facecolor='none',hatch = '//',step='mid')
        #plt.errorbar(bin,hits-h_mc,yerr = yerr_cnn,fmt='ok',label = 'Reco')
        #if plot_cog:
        #    plt.errorbar(bin,(h_cog-h_mc)/lifetime,yerr = yerr_cog,fmt='or',label = 'CoG') 
        #plt.axhline(0,label='test', color= 'red', linestyle = '--')
        #plt.xlabel("R[cm]")
        #plt.grid(True)        
        #plt.xlim(0,3.5)
        #plt.show()
 
    def shist1d(self,vars,labels, density):
        if density:
            for i in range(len(vars)):
                    counts,bins = np.histogram(vars[i],bins = 30)
                    f = 1/np.sum(counts)
                    # plt.step(bins[:-1], counts*f, label = labels[i][0])
                    plt.hist(bins[:-1],bins, weights = f*counts, histtype = 'step', label = labels[i][0])
        else:
            plt.hist(vars,bins = 30, histtype = 'step', label = ["CNN","MC"], density = density)
        plt.xlabel(labels[0][1])
        plt.ylabel(r"Rate [$1\over{s}$]" if density else labels[0][2])
        plt.legend(loc = 'upper left')
        plt.grid(True, axis = 'both')
        plt.show()

    def xyr(self,r,xy, log = False):
        f,axs = plt.subplots(2)
        for i,ax  in enumerate(axs):
            if log != False:
                h,xs,ys,im = ax.hist2d(r,xy[:,0] if i ==0 else xy[:,1],bins=20,cmap = cmocean.cm.dense,norm=mpl.colors.LogNorm())
            else:  
                h,xs,ys,im = ax.hist2d(r,xy[:,0] if i ==0 else xy[:,1],bins=20,cmap = cmocean.cm.dense)
            cbar = plt.colorbar(im,ax=ax)
            cbar.set_label(label = "Events", loc = 'center')
            ax.set_xlabel("R [cm]")
            ax.set_ylabel("X [cm]" if i==0 else "Y [cm]")
        plt.show()

    def hist2d(self,var1,var2,ranges, labels, dig = False, mode = None, pars=None,):
        n,xedges,yedges,_ =plt.hist2d(var1,var2, bins = 30,range=ranges, cmap = cmocean.cm.dense)
        cbar = plt.colorbar()
        cbar.set_label(labels[0])
        plt.xlabel(labels[1])
        plt.ylabel(labels[2])
        if mode == 'sigmas':
            for i in range(1,3):
                rx,ry = i*pars[0][-1], i*pars[1][-1]
                x0,y0 = pars[0][0],pars[1][0]
                print(rx,ry)
                x,y=self.m.helper.ellipse(rx,ry,0,x0,y0)
                plt.plot(x,y,"r--",ms=1,linewidth=1.0)
                plt.annotate('{0}$\sigma$'.format(str(i)), xy=(x0+rx/2,y0+ry/2), xycoords='data',xytext=(10, 10),
                            textcoords='offset points',horizontalalignment='right',
                            verticalalignment='bottom',fontsize=10, color = 'red')
        plt.show()
        if dig:
            m,s = self.m.helper.digitize(var2,xedges,var1)
            plt.errorbar(xedges[1:-4],m[1:-4],yerr=s[1:-4], fmt='o', color = 'k')
            plt.xlabel(labels[1])
            plt.ylabel(labels[2])  
            plt.grid(True, axis = 'both')
            if mode == 'exp':
                popt, pcov = curve_fit(self.m.helper.exp,xedges[1:-4],m[1:-4])
                plt.plot(xedges[1:-4], self.m.helper.exp(xedges[1:-4], *popt), label = r'$N_0 e^{-t_d\over{\tau}}$', color = 'r')
                plt.legend(loc = 'best')
                print("Tau from fit [us] = ", popt[1])
            plt.show()
            return popt[1]

    def draw_s2_top(self,event,pes,pos,cog_pos = None):
        fig, ax = plt.subplots()
        sipm_ids = []
        i = 0
        patches = []
        ws = event[:,:,0].flatten()
        colors = np.asarray(ws)
        for y in self.sipm_ys:
            for x in self.sipm_xs:
              rect = mpatches.Rectangle ((x-self.sipm_w/2,y-self.sipm_h/2),self.sipm_w,self.sipm_h) # bottom left corners
              patches.append(rect)
              i += 1
        p = PatchCollection(patches, edgecolor = 'k',alpha = 0.8, cmap = cmocean.cm.dense)
        p.set_array(np.array(colors))
        ax.add_collection(p)    
        cbar = plt.colorbar(p)
        cbar.set_label('PE')
        plt.xlabel('X [cm]', loc = 'center')
        plt.ylabel('Y [cm]', loc = 'center')
        plt.xlim(-2.5,2.5)
        plt.ylim(-2.5,2.5)
        plt.grid(False)
        plt.scatter(pos[0],pos[1], s = 80, c = 'red', marker = '*',label="Reco")
        if cog_pos!=None:
            plt.scatter(cog_pos[0],cog_pos[1], s = 80, c = 'green', marker = '*',label="CoG")
        plt.legend()
        plt.show()

    def tuned_hist(self,xy1,xy2):
        diff_x , diff_y = xy1[:,0]-xy2[:,0], xy1[:,1]-xy2[:,1],
        nx, binsx,_  = plt.hist(diff_x,bins=40,range=[-0.25,0.25],histtype='stepfilled',alpha = 0.5,label= 'X',color = 'blue')
        ny, binsy, _ = plt.hist(diff_y,bins=40,range=[-0.25,0.25],histtype='stepfilled',alpha = 0.5,label= 'Y',color = 'red')
        bin_centers_x = binsx[:-1] + np.diff(binsx) / 2
        bin_centers_y = binsy[:-1] + np.diff(binsy) / 2
        popt_x, _ = curve_fit(self.gauss, bin_centers_x,nx,maxfev=1000)
        popt_y, _ = curve_fit(self.gauss, bin_centers_y,ny,maxfev=1000)
        xs_x = np.linspace(binsx[0], binsx[-1], 1000)
        xs_y = np.linspace(binsy[0], binsy[-1], 1000)
        plt.plot(xs_x,self.gauss(xs_x, *popt_x), 'r-') 
        plt.plot(xs_y,self.gauss(xs_y, *popt_y), 'b-') 
        plt.xlabel("Reconstructed - Tuned [cm]") 
        plt.ylabel("Events")
        plt.legend(loc='best')
        plt.grid(True)
        plt.xlim(-0.5,0.5)
        plt.show()
        print(abs(popt_x[-1]), abs(popt_y[-1]))

    def hits3d(self,hits):
        fig = plt.figure()         
        ax = fig.add_subplot(111, projection='3d')
        xpos, ypos = np.meshgrid(self.sipm_xs-self.sipm_w/2,self.sipm_ys-self.sipm_h/2)
        w, d = self.sipm_xs[1]-self.sipm_xs[0], self.sipm_ys[1]-self.sipm_ys[0]
        h , edges  = np.histogram(np.argmax(np.reshape(hits,(len(hits),24)),axis=1), bins = 24)
        # h  = order_3dhits(h)
        xpos, ypos, zpos = xpos.flatten(), ypos.flatten(), np.zeros_like(xpos).flatten()
        cmap = cmocean.cm.dense
        ax.bar3d(xpos, ypos, zpos, w, d, h, shade = True, color = 'paleturquoise', edgecolor='teal', alpha = 0.8)
        plt.xlabel("X [cm]")
        plt.ylabel("Y [cm]")
        # plt.gca().invert_yaxis()
        plt.show()

    def spatial_dist(self,x,y,bins=12,log = False,norm=None):
        H,xedges,yedges =  np.histogram2d(x,y,bins = bins, range =  [[-self.max_x,self.max_x],[-self.max_y,self.max_y]])
        X,Y = np.meshgrid(xedges,yedges)
        H = H.T if norm==None else np.divide(H.T,norm)
        if log:
            plt.pcolormesh(X,Y,H,cmap=cmocean.cm.dense, norm=mpl.colors.LogNorm())
            #hits, xedges, yedges, im = plt.hist2d(x,y, bins = bins,range = [[-self.max_x,self.max_x],[-self.max_y,self.max_y]], cmap = cmocean.cm.dense,norm=mpl.colors.LogNorm())
        else:
            plt.pcolormesh(X,Y,H,cmap=cmocean.cm.dense)
            #hits, xedges, yedges, im = plt.hist2d(x,y, bins = bins,range = [[-self.max_y,self.max_y],[-self.max_y,self.max_y]], cmap = cmocean.cm.dense)
        cbar = plt.colorbar()
        cbar.set_label("Events" if norm==None else "Rate [Events/sec]")    
        plt.xlabel("X [cm]")
        plt.ylabel("Y [cm]")
        plt.xlim(-self.max_x,self.max_x)
        plt.ylim(-self.max_y,self.max_y)
        plt.savefig('save/xydist.png',dpi=400)
        plt.show()

    def xydist_residuals(self,xy_reco,xy_reco_tuned,bins):         
        x_devs,y_devs = np.abs(xy_reco[:,0]-xy_reco_tuned[:,0]), np.abs(xy_reco[:,1]-xy_reco_tuned[:,1])
        x,y   = np.linspace(-2.5,2.5,bins),np.linspace(-2.5,2.5,bins)
        xs,ys = np.meshgrid(x,y)
        ret   = binned_statistic_2d(xy_reco[:,0],xy_reco[:,1],[x_devs,y_devs], statistic = 'mean',bins = [x,y]) 
        devs  = np.average(ret.statistic,axis=0) 
        im = plt.pcolormesh(xs,ys,devs,cmap=cmocean.cm.dense)
        cbar = plt.colorbar(im)
        cbar.set_label(label = "True - Reco [cm]", loc = 'center')
        plt.xlabel("X [cm]")
        plt.ylabel("Y [cm]")
        plt.savefig('save/xysigma.png',dpi=400)  
        plt.show()

    def gauss(self,x, a, x0, sigma):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


'''
Add the following if wanting to include residuals plot, doubtful
fig.add_axes([0.1,0.1,0.8,0.2])
#plt.fill_between(bin,-yerr_mc,yerr_mc,facecolor='none',hatch = '//',step='mid')
#plt.errorbar(bin,hits-h_mc,yerr = yerr_cnn,fmt='ok',label = 'Reco')
#if plot_cog:
#    plt.errorbar(bin,(h_cog-h_mc)/lifetime,yerr = yerr_cog,fmt='or',label = 'CoG') 
#plt.axhline(0,label='test', color= 'red', linestyle = '--')
#plt.xlabel("R[cm]")
#plt.grid(True)        
#plt.xlim(0,3.5)
#plt.show() 
'''