import numpy as np

from lib.manager import Manager
m = Manager() 

class RecoXY():
    xs, ys = np.linspace(-2.5,2.5,30), np.linspace(-2.5,2.5,30)
    rs    = np.sqrt(xs**2 + ys**2)
    un_xs = [float(i) for i in m.config('un','un_xs','str').split(',')]
    un_ys = [float(i) for i in m.config('un','un_ys','str').split(',')] 
    un_rs = np.sqrt(np.square(np.divide(xs,rs)*un_xs)+np.square(np.divide(ys,rs))*un_ys)
    un_xs_mc = [float(i) for i in m.config('un','un_xs_mc','str').split(',')] 
    un_ys_mc = [float(i) for i in m.config('un','un_ys_mc','str').split(',')] 
    un_rs_mc = [float(i) for i in m.config('un','un_rs_mc','str').split(',')] 

    def __init__(self,file,pars = [[],[]],bg = False):
        self.tree     = m.helper.read_tree(file)
        self.branches = m.helper.read_branches(self.tree)
        self.pars     = pars
        self.model    = m.config('file','path','str')+m.config('cnn','model','str') 
        self.lifetime = m.config('run','lifetime_bg','int')*60 if bg else m.config('run','lifetime','int')*60 

        # prepare data 
        self.s1, self.s2, self.tdrift, self.tdrift_s2, self.cut_ncl = self.get_s1ns2(self.branches) 
        self.tau, ps1, ps2 = m.plotter.inspect(self.s1,self.s2,self.tdrift,self.tdrift_s2,self.pars)
        self.s2_top          = self.cuts_n_s2top(self.cut_ncl,self.pars)
        if len(self.pars[0])==0:
            self.pars[0],self.pars[1] = ps1,ps2
        #self.s2_top_corr     = self.s2_corr(self.s2_top,self.tdrift_s2)

        # reconstruct and get mc data 
        self.xy_reco, self.hits, self.hits_n = self.reco_xy(self.s2_top)
        self.r_reco  = (self.xy_reco[:,0]**2 + self.xy_reco[:,1]**2)**0.5 
        self.xy_mc = m.helper.read_mc(m.config('file','path','str')+m.config('file','fname_mc','str'))
        self.r_mc  = (self.xy_mc[:,0]**2 + self.xy_mc[:,1]**2)**0.5 
        self.lifetime_mc = (len(self.xy_mc)*self.lifetime)/len(self.xy_reco)
       
        # cog to compare, should be drastically different around the edges 
        #self.xy_cog, _,_ = self.reco_xy(self.s2_top_corr,cnn=False)
        #self.r_cog  = (self.xy_cog[:,0]**2 + self.xy_cog[:,1]**2)**0.5
        
        # some plots
        for i in range(0):
            #m.plotter.draw_s2_top(self.hits_n[i], 1, [self.xy_reco[i,0],self.xy_reco[i,1]],[self.xy_cog[i,0],self.xy_cog[i,1]]) 
            m.plotter.draw_s2_top(self.hits_n[i], 1, [self.xy_reco[i,0],self.xy_reco[i,1]]) 
        self.evts     = len(self.xy_reco) if (len(self.xy_reco) < len(self.xy_mc)) else len(self.xy_mc)
        bins          = 30

        m.plotter.coord_reco(self.xy_reco[:,0], self.xy_mc[:,0], bins, (-2.5,2.5), ["Reco","X [cm]","Rate [Events/sec]" ],self.un_xs,self.un_xs_mc)
        m.plotter.coord_reco(self.xy_reco[:,1], self.xy_mc[:,1], bins, (-2.5,2.5), ["Reco","Y [cm]", "Rate [Events/sec]" ],self.un_ys,self.un_ys_mc)
        m.plotter.spatial_dist(self.xy_reco[:,0],self.xy_reco[:,1], bins=12,log = False,norm=self.lifetime) 
        m.plotter.spatial_dist(self.xy_reco[:,0],self.xy_reco[:,1], bins=24,log = False,norm=self.lifetime) 
        m.plotter.coord_reco(self.r_reco,self.r_mc,bins,(0,3.5),["Reco","R [cm]","Rate [Events/sec]" ],self.un_rs,self.un_rs_mc) 

        if not bg:
            # if fine tunning with reconstructed data
            if m.config('cnn','train_reco','bool'):
                self.model, _          = m.cnn.train(self.hits,self.xy_reco)
            else:
                self.model             = m.config('file','path','str')+m.config('cnn','model_tr' ,'str') 
            self.xy_reco_tuned,_,_ = self.reco_xy(self.s2_top) 
            self.r_reco_tuned  = (self.xy_reco_tuned[:,0]**2 + self.xy_reco_tuned[:,1]**2)**0.5         
        
            bins = 30 
            for i in range(0):
                m.plotter.draw_s2_top(self.hits_n[i], 1, [self.xy_reco[i,0],self.xy_reco[i,1]])        
                m.plotter.draw_s2_top(self.hits_n[i], 1, [self.xy_reco_tuned[i,0],self.xy_reco_tuned[i,1]])        

            m.plotter.tuned_hist(self.xy_reco,self.xy_reco_tuned)            
            m.plotter.spatial_dist(self.xy_reco_tuned[:,0],self.xy_reco_tuned[:,1], bins=bins-10,log = False) 
            m.plotter.xydist_residuals(self.xy_reco,self.xy_reco_tuned,16)
            m.plotter.xydist_residuals(self.xy_reco,self.xy_reco_tuned,31)

    def get_s1ns2(self,branches):
        cut_ncl = (branches["ncl_s2"] == 1) & (branches["ncl_s1"] == 1)
        s1_q = branches["cl_q"][cut_ncl][:,0]
        s2_q = branches["cl_q"][cut_ncl][:,1]
        cl_startt = branches["cl_startt"][branches["ncl"]>1]      # all events with nclus > 1
        tdrift = (cl_startt[:,1] - cl_startt[:,0])*2e-3
        cl_startt_s2 = branches["cl_startt"][cut_ncl]             # all events with 1 s1 & 1 s2
        tdrift_s2 = (cl_startt_s2[:,1] - cl_startt_s2[:,0])*2e-3    

        return s1_q, s2_q, tdrift, tdrift_s2, cut_ncl

    def cuts_n_s2top(self,cut_ncl,pars):
        print("###### CUTS #######")
        print((pars[0][0]-2*pars[0][1]), "< S1 [PE] < ", (pars[0][0]+2*pars[0][1]))
        print((pars[1][0]-2*pars[1][1]), "< S2 [PE] < ", (pars[1][0]+2*pars[1][1]))
        cut_s1 = ((pars[0][0]-2*pars[0][1]) < self.s1) & (self.s1<(pars[0][0]+2*pars[0][1])) 
        cut_s2 = ((pars[1][0]-2*pars[1][1]) < self.s2) & (self.s2<(pars[1][0]+2*pars[1][1])) 
        cut_tdrift = (self.tdrift_s2>10) & (self.tdrift_s2<60)

        cuts        = (cut_s1) & (cut_s2) & (cut_tdrift)
        pk_t        = self.branches["pk_t"][cut_ncl][cuts]
        pk_npe      = self.branches["pk_npe"][cut_ncl][cuts] 
        pk_ch       = self.branches["pk_ch"][cut_ncl][cuts]
        cl_startt   = self.branches["cl_startt"][cut_ncl][cuts]
        cl_endt     = self.branches["cl_endt"][cut_ncl][cuts]

        print("Getting S2 top per channel...")
        s2_top_chan, _ = m.helper.get_s2_top_chan(pk_t,pk_npe,pk_ch,cl_startt,cl_endt)

        return s2_top_chan    
    
    def s2_corr(self,s2_top,tdrift):
        s2_top_chan_corr = np.zeros_like(s2_top)
        for i in range(len(s2_top)):
            s2_top_chan_corr[i] = s2_top[i] / np.exp(-tdrift[i]/self.tau) 

        return s2_top_chan_corr
    
    def reco_xy(self,s2_top,cnn=True,mc=False):
        hits, hits_n   = m.helper.format_data(s2_top,cnn,mc) 
        if cnn:
            xy             = m.cnn.predict_pos(self.model,hits)
        else:
            xy             = m.helper.center_of_gravity(hits_n)
        return xy, hits, hits_n


###########################
########## MAIN ###########
###########################
run_am   = RecoXY(m.config('file','path','str')+m.config('file','fname','str'),[[665,100],[12500,2500]])
#evts     = run_am.evts
#xy_reco  = run_am.xy_reco
#xy_mc    = run_am.xy_mc
#r_reco   = run_am.r_reco
#r_mc     = run_am.r_mc
#cut_pars = run_am.pars

#run_bg   = RecoXY(m.config('file','path','str')+m.config('file','fname_bg',"str"),pars=cut_pars, bg=True)
#xy_bg    = run_bg.xy_reco
#r_bg     = run_bg.r_reco
      
#m.plotter.coord_reco(xy_reco[:,0], xy_mc[:,0], 30, (-2.5,2.5), [r"$\rm{Am}^{241}$"+" - Bg","X [cm]","Rate [Events/sec]" ],run_am.un_xs,run_am.un_xs_mc,bg=xy_bg[:,0])
#m.plotter.coord_reco(xy_reco[:,1], xy_mc[:,1], 30, (-2.5,2.5), [r"$\rm{Am}^{241}$"+" - Bg","Y [cm]","Rate [Events/sec]" ],run_am.un_xs,run_am.un_xs_mc,bg=xy_bg[:,1])
#m.plotter.coord_reco(r_reco,r_mc, 30, (0,3.5), [r"$\rm{Am}^{241}$"+" - Bg","R [cm]","Rate [Events/sec]" ],run_am.un_xs,run_am.un_xs_mc,bg=r_bg)
