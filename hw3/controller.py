def controller(states,samples=50,alpha=1,lamb=0.1,mu=2):
    ''' simplistic MPC loop '''
    global u,pd
    #shifting 1 timestep
    u[:,:-1]=u[:,1:]
    u[:,-1] =0
    Nhorizon = 10
    u_guess = np.reshape(np.random.multivariate_normal(np.zeros(samples*3*Nhorizon), np.eye(samples*3*Nhorizon)),[samples,3,Nhorizon])
    
    l=lambda x: np.sum(np.square(manipulator.forward_kinematics(x)-pd))
    la= np.apply_along_axis(l, axis=0, arr=states)
    if la.ndim > 0:
        la=np.sum(la[:min(10,len(la))])


    l= la+0.01*np.sum(np.square(u+mu*u_guess),axis=(1, 2), keepdims=True)
    exp =  np.exp((-1/lamb)*l)
    # print(np.max(l),np.min(l))
    num=np.sum(-lamb*exp*u_guess,axis=0)
    den=np.sum(mu*exp,axis=0)
    # print(la)
    # print(l)
    # print(u_guess)
    # print(den)
    u = u - alpha*num/den
    return u[:,0]
