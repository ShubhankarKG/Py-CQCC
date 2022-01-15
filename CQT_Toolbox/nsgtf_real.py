from distutils.log import error
from math import ceil, floor
import sys
import numpy as np

def nsgtf_real(**args):
    nargin = len(args)

    if nargin <= 2:
        error("Not enough input arguments")
        sys.exit(1)
    
    f, g, shift = args[0], args[1], args[2]
    Ls, CH = f.shape

    if Ls==1:
        f = f.T
        Ls = CH
        CH = 1
    
    if CH > Ls:
        print(f"The number of signal channels {CH} is larger than the number of samples per channel {Ls}\n")
        reply = input("Is this correct? ([Y]es, [N]o) ")
        if reply in ['N', 'n', 'No', 'no', '']:
            reply2 = input('Transpose signal matrix? ([Y]es, [N]o) ')
            if reply2 in ['N', 'n', 'No', 'no', '']:
                error("Invalid signal input, terminating program")
                sys.exit(1)
            elif reply2 in ['Y', 'y', 'Yes', 'yes']:
                print("Transposing signal matrix and continuing program exexution")
                f = f.T
                CH, Ls = Ls, CH
        elif reply in ['Y', 'y', 'Yes', 'yes']:
            print("Continuing program execution")
        else:
            error("Invalid input, terminating program")
            sys.exit(1)
        
    N = len(shift)
    M = args[3]
    phasemode = args[4]
    if nargin == 3:
        M = np.zeros((N, 1))
        for i in range(N):
            M[i] = len(g[i])
        
    if max(M.shape) == 1:
        M = M[0] * np.ones((N, 1))
    
    # Some preparation
    f = np.fft.fft(f, axis=0)
    posit = np.cumsum(shift) - shift[0] # Calculate positions from shift vector

    # A small amount of zero-padding might be needed (e.g. for scale frames)
    fill = sum(shift) - Ls
    f = np.concatenate((f, np.zeros((fill, CH))), axis=0)

    Lg = np.array([len(cell) for cell in g])
    N = np.where(posit-floor(Lg/2)<=(Ls+fill)/2)[0][-1]
    c = np.array([])

    # The actual transform
    for i in range(N+1):
        idx_r1 = np.arange(ceil(Lg[i]/2)-1, Lg[i])
        idx = np.array(
            np.append(idx_r1, np.arange(ceil(Lg[i]/2))),
            dtype=int
        )
        win_range = (posit[i] + np.arange(-floor(Lg[i]/2)-1, ceil(Lg[i]/2)-1)) % (Ls+fill) + 1
        win_range = np.array(win_range)

        if M[i] < Lg[i]: # If the number of frequency channels is too small
            # Aliasing is introduced (non-painless case)
            col = ceil(Lg[i]/ M[i])
            temp = np.zeros((col * M[i], CH))
            end = col*M[i]
            idx_list = list(range(end-np.floor(Lg[i]/2)+1-1, end)) + list(range(0,np.ceil(Lg[i]/2)))
            temp[idx_list,:] = f[win_range,:] * g[i][idx]
            temp = np.reshape(temp,(M[i],col,CH))

            c.append(np.squeeze(np.fft.ifft(np.sum(temp,axis=1))))
        else:
            temp = np.zeros((M[i], CH))
            end = int(M[i])

            idx_list = list(range(int(end-np.floor(Lg[i]/2)+1-1), end)) + list(range(0,int(np.ceil(Lg[i]/2))))
            idx_array = np.array(idx_list)
            temp[idx_array,:] = f[win_range] * g[i][idx]

            if phasemode == 'global':
                # Apply frequency mapping (See CQT)
                fsNewBins = M[i]
                fkBins = posit[i]
                displace = fkBins - floor(fkBins/fsNewBins) * fsNewBins
                temp = np.roll(temp, displace)

            c.append(np.fft.ifft(temp))
        
    
    if max(M) == min(M):
        c_list = [c[i][0] for i in range(len(c))]
        c = np.vstack(c_list).astype(c[0].dtype)
        c = np.reshape(c,(M[0],N,CH))

    return c, Ls








