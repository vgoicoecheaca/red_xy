import numpy as np
from numpy.ma.core import concatenate
import uproot as ur
import h5py 
from scipy.optimize import curve_fit
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from math import lgamma, log

class Helper():
    def __init__(self,manager):
        self.m       = manager
        self.nevents = self.m.config('run','nevents','int')
        self.max_x   = self.m.config('run','max_x','float')
        self.max_y   = self.m.config('run','max_y','float')   
        self.sipm_xs = np.linspace(-2.5+(1.25/2),2.5-(1.25/2),4) 
        self.sipm_ys = np.linspace(-2.5+(0.833333/2),2.5-(0.833333/2),6)
        self.sipm_w , self.sipm_h = self.sipm_xs[1] - self.sipm_xs[0], self.sipm_ys[1]-self.sipm_ys[0]
        self.channel_dec_alltpc   = self.m.config('run','channel_dec_alltpc','str').split(',') 
        self.channel_na           = self.m.config('run','channel_na','str').split(',')
        self.channel_map          = self.channel_dec_alltpc if self.m.config('run','channel_map','str') == 'channel_dec_alltpc' else self.channel_na 
        self.channel_config       = self.m.config('run','channel_config','str').split(',')

    def read_tree(self,file): 
        print("Reading tree")
        return ur.open(file)['dstree']

    def read_branches(self,tree,branches = None):
        print("Reading branches...")
        if  branches == None:
            return tree.arrays()[:self.nevents]
        else:
            return tree.arrays([str(branch) for branch in branches],library='np')

    def read_mc(self,file):
        if file[-1]=='t':
            print(file)
            tree = self.read_tree(file)
            branches = self.read_branches(tree, ["dep_x", "dep_y"])
            dep_x = np.concatenate(np.ravel(branches["dep_x"]))
            dep_y = np.concatenate(np.ravel(branches["dep_y"]))
            cut = (dep_x >= -2.5) & (dep_x <=2.5) & (dep_y >= -2.5) & (dep_y <=2.5) 
            return np.vstack([dep_x[cut],dep_y[cut]]).T
        else:
            return self.read_hdf5(file)

    def read_hdf5(self,file):
        print("Reading", file, "hdf5 file")
        f = h5py.File(file,'r') 
        if len(list(f.keys()))==3:
            return np.array([f['xTrue'][:self.nevents],f['yTrue'][:self.nevents]], dtype =np.float32).T, np.array(f['pe_pdm'][:self.nevents,:],dtype=np.float32)
        else:
            return np.array([f['x'][:self.nevents],f['y'][:self.nevents]], dtype =np.float32).T, None

    def map_channel(self,event, chs):
          nx_sipms, ny_sipms = 4,6
          hits = np.zeros((nx_sipms*ny_sipms),dtype = np.float32)
          for i in range(len(chs)):
            if chs[i][0] == 'F':
              continue
            if chs[i] in self.channel_config:
              hits[self.channel_config.index(chs[i])] = event[i] 
            else:
              hits[i] = 0  
          return hits

    def normalize(self,s2,pes,mode):
        if mode == 'norm':
            return s2 / pes[:, np.newaxis] 
        elif mode == 'minmax':
            return (s2 - s2.min(axis=1).reshape(s2.shape[0],1)) / (s2.max(axis=1).reshape(s2.shape[0],1) - s2.min(axis=1).reshape(s2.shape[0],1))
        elif mode == 'znorm': 
            return (s2 - s2.mean(axis=1).reshape(s2.shape[0],1)) / s2.std(axis=1).reshape(s2.shape[0],1)    

    def get_s2_top_chan(self,pk_t,pk_npe,pk_ch,cl_startt,cl_endt):
        pe_ch = np.zeros((len(pk_t),24),dtype = np.float32)
        test = []
        for event in range(len(pk_t)):
            if event % 10000==0:
                print(event,"/",len(pk_t))
            try:
                t0 = cl_startt[event][1]
                t1 = cl_endt[event][1]
                mask = (pk_t[event] > t0) & (pk_t[event] < t1)   
            except:
                ValueError("Value too large")
                continue 
            pnpe = np.array(pk_npe[event][mask])  # this is the same as cl_p for s2
            ch = np.array(np.unique(pk_ch[event][mask]))
            if len(ch) == 0:
                continue
            ch_w = np.zeros(ch.shape,dtype=np.float)
            for i,c in np.ndenumerate(ch):
                ch_w[i] = sum(pnpe[pk_ch[event][mask]==c])
            if ch_w.shape[0] == 28:
                pe_ch[event] = self.map_channel(ch_w,self.channel_map)

        return pe_ch[~np.all(pe_ch == 0, axis=1)], ~np.all(pe_ch == 0, axis=1) 

    def gauss_fit(self,n, bins):
            bin_centers = bins[:-1] + np.diff(bins) / 2
            bounds = ([100,bins[np.argmax(n)]-200, 0], [50000,bins[np.argmax(n)]+200, 1000])
            popt, _ = curve_fit(self.gauss, bin_centers,n ,maxfev=1000)
            lwr , upr = popt[-2]-(2*popt[-1]), popt[-2]+(2*popt[-1])
            xs = np.linspace(lwr, upr, 1000)
            return xs, self.gauss(xs, *popt), [popt[-2],popt[-1]]

    def format_data(self,s2_top_chan,cnn=True,mc=False):
        pes = np.sum(s2_top_chan,axis=1)
        if cnn or mc:
            print("Formating data into CNN model input...")
            hits = np.zeros((len(s2_top_chan),6,4,1), dtype = np.float32) 
            hits_n = np.zeros((len(s2_top_chan),6,4,1), dtype = np.float32)  
            s2_top_chan_norm = self.normalize(s2_top_chan,pes,"znorm") 
        else:
            print("Formating data into CoG model input...")
            hits = np.zeros((len(s2_top_chan),6,4,1), dtype = np.float32) 
            hits_n = np.zeros((len(s2_top_chan),6,4,1), dtype = np.float32)  
            for i in range(len(self.sipm_ys)): hits[:,i,:,2] = self.sipm_ys[i] 
            for i in range(len(self.sipm_xs)): hits[:,:,i,1] = self.sipm_xs[i]   
            hits_n[:,:,:,1] = hits[:,:,:,1] 
            hits_n[:,:,:,2] = hits[:,:,:,2]
        for i in range(len(s2_top_chan)): 
            if mc:
                hits[i,:,:,0]   = np.reshape(s2_top_chan_norm[i], (6,4))
            else:
                if cnn:
                    hits[i,:,:,0]  = np.flipud(np.reshape(s2_top_chan_norm[i],(6,4)))
                    hits_n[i,:,:,0] = np.flipud(np.reshape(s2_top_chan[i],(6,4)))
                else:
                    hits[i,:,:,0]  = np.flipud(np.reshape(s2_top_chan[i],(6,4)))
                    hits_n[i,:,:,0] = np.flipud(np.reshape(s2_top_chan[i],(6,4))) 
        return hits,hits_n

    def conv(self,hits,bin_edges):
        bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2
        res, smear = np.zeros_like(bin_centers), np.array([])
        fres = np.poly1d([float(i) for i in self.m.config('un','res_pars','str').split(',')])
        for i in range(len(bin_centers)):
            res[i]        = fres(bin_centers[i]) 
            vals = np.random.normal(bin_centers[i], np.abs(bin_centers[i]*res[i]), int(hits[i]))
            smear =np.append(smear,vals.flatten())
        return smear.flatten()

    def order_3dhits(self,hits):
        dz_0 = np.copy(hits)
        idx, vals  = [6,7,18,19], [dz_0[4],dz_0[5],dz_0[16],dz_0[17]]
        np.put(hits, idx, vals)
        idx, vals  = [4,5,16,17], [dz_0[6],dz_0[7],dz_0[18],dz_0[19]]
        np.put(hits, idx, vals)
        part = np.roll(hits[:16],4)
        hits[:-8] = part
        hits = np.reshape(hits, (6,4))
        hits[[0,1]] = hits[[1,0]]

        return hits.flatten()

    def ellipse(self,ra,rb,ang,x0,y0):
        xpos,ypos=x0,y0
        radm,radn=ra,rb
        an=ang
        co,si=np.cos(an),np.sin(an)
        the=np.linspace(0,2*np.pi,100)
        X=radm*np.cos(the)*co-si*radn*np.sin(the)+xpos
        Y=radm*np.cos(the)*si+co*radn*np.sin(the)+ypos

        return X,Y

    def digitize(self,data,edges,data_idxs):
        m , s = np.zeros((len(edges))), np.zeros((len(edges)))
        d = np.digitize(data_idxs,edges)
        for i in range(len(edges)):
            m[i]  = np.mean(data[np.where(d == i)[0]])
            s[i]  = np.sqrt(m[i]/len(np.where(d ==i)[0]))

        return m,s 

    def center_of_gravity(self,hits):
        '''Only using 4 channels (2 in x and 2 in y)'''
        xy = np.zeros((len(hits),2))
        for i in range(len(xy)):
            m  = np.unravel_index(np.argmax(hits[i,:,:,0]), (6,4)) 
            edge_ids = [0,5,3] 
            if True in np.in1d(m,edge_ids):
                y_i, x_i = m[0] // 2, 1 if m[1] >= 2 else 0   
                quadrant = hits[i,int(y_i*2):int(y_i*2)+2,int(x_i*2):int(x_i*2)+2,:].reshape(4,3)                    
            else:
                y_low,y_up,x_low,x_up = m[0]-1, m[0]+2, m[1]-1, m[1]+2
                quadrant = hits[i,y_low:y_up,x_low:x_up,:].reshape(9,3)  
            xy[i,:] = np.average(quadrant[:,1:], axis=0, weights=quadrant[:,0])
        return xy
     
    def line(self,x,m,b): return m*x + b 

    def gauss(self,x, a, x0, sigma): return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    def exp(self,x, a, b): return a * np.exp(-(x/b))

#hits[i,:,:,0] = np.flipud(np.reshape(s2_topgt_chan[i],(6,4)))
#hits[i,:,:,0] = pad(np.flipud(np.reshape(s2_top_chan[i],(6,4))),[[1,1],[1,1]],"REFLECT")
       