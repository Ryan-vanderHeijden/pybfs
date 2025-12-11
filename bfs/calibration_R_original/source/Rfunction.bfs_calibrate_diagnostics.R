# Modified bfs_calibrate with diagnostic output
# This is a copy of Rfunction.bfs_calibrate.R with added diagnostic print statements

#CALIBRATE A SITE
bfs_calibrate_diagnostics=function(tmp.site,tmp.area,tmp.q,dys) {

#RUN FLOW METRICS FUNCITON TO GENERATED VARIABLES IN flow VECTOR
  flow=flow_metrics(tmp.q,timestep='day',fr4rise=0.05)

#ONLY PROCEED WITH CALIBRATION IF flow_metrics RETURNS VALUES FOR ALL PARAMETER
  if(any(c(is.na(flow),flow==-Inf))) {bf_params=NA; bff=NA; ci_table=NA; bfs_out=NA} else {
    Qthresh=flow[1]
    Rs=flow[2]
    Rb1=flow[3]
    Rb2=flow[4]
    Prec=flow[5]
    Frac4Rise=flow[6]

    Qmean=mean(tmp.q[tmp.q>=0],na.rm=TRUE)

    RbI=rb10[1]
    RbS=rb10[2]

#INITIALIZE PARAMETERS WITH NOMINAL ESTIMATES
    Lb=2*(tmp.area/2)^0.5
    Wb=tmp.area/Lb/10
    Ws=Wb/2
    POR=0.15

    ALPHA=0.01
    BETA=1
    X1=1/ALPHA

    tmp=ini_params(tmp.area,Lb,X1,Wb,POR,BETA,Rb1,tmp.q)
    Lb=as.numeric(tmp$Lb)
    Wb=as.numeric(tmp$Wb)
    Kb=as.numeric(tmp$Kb)

    Ks=(1-exp(Rs))*POR*(3/4*Ws)/ALPHA
    Kz=10*Qmean/(Lb*Wb)

    basin_char=c(tmp.area,Lb,X1,Wb,POR)
    gw_hyd=c(ALPHA,BETA,Ks,Kb,Kz)

    Error=bfs(tmp.q,dys,timestep='day',error_basis='base',basin_char,gw_hyd,flow)
    BFF=sum(bfs_out$Baseflow.L3,na.rm=T)/sum(bfs_out$Qob.L3,na.rm=T)
    
    # DIAGNOSTIC: Initial parameters (before Step 1)
    cat("\n=== DIAGNOSTIC: INITIAL PARAMETERS (before Step 1) ===\n")
    cat(sprintf("Lb=%.6f, X1=%.6f, Wb=%.6f, POR=%.6f\n", Lb, X1, Wb, POR))
    cat(sprintf("ALPHA=%.6f, BETA=%.6f\n", ALPHA, BETA))
    cat(sprintf("Ks=%.6f, Kb=%.6f, Kz=%.6f\n", Ks, Kb, Kz))
    cat(sprintf("Error=%.6f, BFF=%.6f\n", Error, BFF))
    
    bf_params=data.frame(tmp.site,tmp.area,Lb,X1,Wb,POR,ALPHA,BETA,Ks,Kb,Kz,Qthresh,Rs,Rb1,Rb2,Prec,Frac4Rise,Error,BFF)
#########################################################################################################
#STEP 1. CALIBRATE ASSUMING BETA = 1
#########################################################################################################
    cat("\n  Step 1: Initial calibration (beta=1)...\n")
    X=c(Lb,Wb,ALPHA,Ks,Kb,Kz)

    LOGX=log(X,10)

    tmp=optim(LOGX,cal_initial,tmp.q=tmp.q,dys=dys,timestep='day',error_basis='base',basin_char=basin_char,gw_hyd=gw_hyd,flow=flow,qmean=Qmean,
      control=list(maxit=1000,parscale=LOGX,reltol=0.01))

    # DIAGNOSTIC: After Step 1 cal_initial
    cat("\n=== DIAGNOSTIC: STEP 1 - AFTER cal_initial ===\n")
    cat(sprintf("Optimization success: %s, iterations: %d, final objective: %.6f\n", 
        ifelse(all(is.finite(tmp$par)), "TRUE", "FALSE"), tmp$counts[1], tmp$value))
    if(all(is.finite(tmp$par))) {
      cat(sprintf("Optimized params (log10): %.6f %.6f %.6f %.6f %.6f %.6f\n", 
          tmp$par[1], tmp$par[2], tmp$par[3], tmp$par[4], tmp$par[5], tmp$par[6]))
      cat(sprintf("Optimized params (linear): Lb=%.6f, Wb=%.6f, ALPHA=%.6f, Ks=%.6f, Kb=%.6f, Kz=%.6f\n",
          10^tmp$par[1], 10^tmp$par[2], 10^tmp$par[3], 10^tmp$par[4], 10^tmp$par[5], 10^tmp$par[6]))
    }

#UPDATE PARAMETERS AND PARAMETER SETS
    if(all(is.finite(tmp$par))){
      Lb=10^tmp$par[1];Wb=10^tmp$par[2];ALPHA=10^tmp$par[3];Ks=10^tmp$par[4];Kb=10^tmp$par[5];Kz=10^tmp$par[6]
      basin_char=c(tmp.area,Lb,X1,Wb,POR)
      gw_hyd=c(ALPHA,BETA,Ks,Kb,Kz)
      
      X=c(Lb,Wb,ALPHA,Ks)

      LOGX=log(X,10)

      tmp=optim(LOGX,cal_surface,tmp.q=tmp.q,dys=dys,timestep='day',error_basis='base',basin_char=basin_char,gw_hyd=gw_hyd,flow=flow,
        control=list(maxit=1000,parscale=LOGX,reltol=0.01))
      
      # DIAGNOSTIC: After Step 1 cal_surface
      cat("\n=== DIAGNOSTIC: STEP 1 - AFTER cal_surface ===\n")
      cat(sprintf("Optimization success: %s, iterations: %d, final objective: %.6f\n",
          ifelse(all(is.finite(tmp$par)), "TRUE", "FALSE"), tmp$counts[1], tmp$value))
      if(all(is.finite(tmp$par))) {
        cat(sprintf("Optimized params (log10): %.6f %.6f %.6f %.6f\n",
            tmp$par[1], tmp$par[2], tmp$par[3], tmp$par[4]))
        cat(sprintf("Optimized params (linear): Lb=%.6f, Wb=%.6f, ALPHA=%.6f, Ks=%.6f\n",
            10^tmp$par[1], 10^tmp$par[2], 10^tmp$par[3], 10^tmp$par[4]))
      }
    }

    if(all(is.finite(tmp$par))){
      Lb=10^tmp$par[1];Wb=10^tmp$par[2];ALPHA=10^tmp$par[3];Ks=10^tmp$par[4]
      basin_char=c(tmp.area,Lb,X1,Wb,POR)
      gw_hyd=c(ALPHA,BETA,Ks,Kb,Kz)

      Error=bfs(tmp.q,dys,timestep='day',error_basis='base',basin_char,gw_hyd,flow)
      tmp.bf=bfs_out$Baseflow.L3
      tmp.ov=bfs_out$Baseflow.L3>bfs_out$Qob.L3
      tmp.ov[is.na(tmp.ov)]=FALSE
      tmp.bf[tmp.ov]=bfs_out$Qob.L3[tmp.ov]
      BFF=sum(tmp.bf[!is.na(bfs_out$Qob.L3)])/sum(bfs_out$Qob.L3[!is.na(bfs_out$Qob.L3)])
      
      # DIAGNOSTIC: After Step 1 final BFS run
      cat("\n=== DIAGNOSTIC: STEP 1 - AFTER final BFS run ===\n")
      cat(sprintf("Lb=%.6f, X1=%.6f, Wb=%.6f, POR=%.6f\n", Lb, X1, Wb, POR))
      cat(sprintf("ALPHA=%.6f, BETA=%.6f\n", ALPHA, BETA))
      cat(sprintf("Ks=%.6f, Kb=%.6f, Kz=%.6f\n", Ks, Kb, Kz))
      cat(sprintf("Error=%.6f, BFF=%.6f\n", Error, BFF))
    }

    bf_params=rbind(bf_params,data.frame(tmp.site,tmp.area,Lb,X1,Wb,POR,ALPHA,BETA,Ks,Kb,Kz,Qthresh,Rs,Rb1,Rb2,Prec,Frac4Rise,Error,BFF))
##################################################################################################################################
#STEP 2. CALIBRATE NON-LINEAR BASEFLOW FUNCTION FOR RECESSION RATES AT BETA = 1 TO 20, PICK BEST BETA AND RE-CALIBRATE BF_SEP
##################################################################################################################################
    cat("\n  Step 2: Calibrating non-linear baseflow function (testing beta values)...\n")
    tmp.out=array(dim=c(0,6))

    n=0
    b=0.5
    continue=TRUE
    
    while(continue) {b=b+0.1

      X1=(Wb * Kb * b^2 / ((2*b - 1) * Qmean) * (Lb/2)^(2 * b-1))^(1/(2*b))
      
      if(is.finite(X1)) {X=c(Lb,X1,Wb,Kb)

        tmp=optim(X,cal_basetable,b=b,params=c(tmp.area,POR,Qmean,Qthresh,RbI,RbS),tmp.q=tmp.q,
          control=list(maxit=1000,parscale=X,reltol=0.01))

        if(all(is.finite(tmp$par))) {n=n+1
          tmp.out=rbind(tmp.out,c(tmp$par[1],tmp$par[2],tmp$par[3],b,tmp$par[4],tmp$value))
          cat(sprintf("    Tested beta=%.1f, objective=%.4f\n", b, tmp$value))
        }}

      if(b>10) {if(abs((tmp$value-tmp.out[n,6])/tmp$value)<0.001) {continue=FALSE}}
      if(b==20) {continue=FALSE}}

    X=tmp.out[match(min(tmp.out[,6]),tmp.out[,6]),1:5]

    Lb=X[1];X1=X[2];Wb=X[3]
    basin_char=c(tmp.area,X[1],X[2],Wb,POR)

    BETA=X[4];Kb=X[5]
    cat(sprintf("    Selected best beta=%.3f with objective=%.4f\n", BETA, min(tmp.out[,6])))
    gw_hyd=c(ALPHA,BETA,Ks,Kb,Kz)

    X=c(Lb,Wb,Kb,Kz)

    cat("    Optimizing base parameters...\n")
    tmp=optim(X,cal_base,tmp.q=tmp.q,dys=dys,timestep='day',error_basis='base',basin_char=basin_char,gw_hyd=gw_hyd,flow=flow,
      control=list(maxit=1000,parscale=X,reltol=0.01))
    tmp$par[tmp$par<0]=NA

    # DIAGNOSTIC: After Step 2 cal_base
    cat(sprintf("\n    === DIAGNOSTIC: STEP 2 - AFTER cal_base (beta=%.3f) ===\n", BETA))
    cat(sprintf("    Optimization success: %s, iterations: %d, final objective: %.6f\n",
        ifelse(all(is.finite(tmp$par)), "TRUE", "FALSE"), tmp$counts[1], tmp$value))
    if(all(is.finite(tmp$par))) {
      cat(sprintf("    Optimized params: Lb=%.6f, Wb=%.6f, Kb=%.6f, Kz=%.6f\n",
          tmp$par[1], tmp$par[2], tmp$par[3], tmp$par[4]))
    }

#UPDATE PARAMETERS AND PARAMETER SETS
    if(all(is.finite(tmp$par))){
      Lb=tmp$par[1];Wb=tmp$par[2];Kb=tmp$par[3];Kz=tmp$par[4]
      
      X=c(Lb,Wb,ALPHA,Ks)
      LOGX=log(X,10)
      tmp=optim(LOGX,cal_surface,tmp.q=tmp.q,dys=dys,timestep='day',error_basis='base',basin_char=basin_char,gw_hyd=gw_hyd,flow=flow,
        control=list(maxit=1000,parscale=LOGX,reltol=0.01))

      # DIAGNOSTIC: After Step 2 cal_surface
      cat(sprintf("\n    === DIAGNOSTIC: STEP 2 - AFTER cal_surface (beta=%.3f) ===\n", BETA))
      cat(sprintf("    Optimization success: %s, iterations: %d, final objective: %.6f\n",
          ifelse(all(is.finite(tmp$par)), "TRUE", "FALSE"), tmp$counts[1], tmp$value))
      if(all(is.finite(tmp$par))) {
        cat(sprintf("    Optimized params (log10): %.6f %.6f %.6f %.6f\n",
            tmp$par[1], tmp$par[2], tmp$par[3], tmp$par[4]))
        cat(sprintf("    Optimized params (linear): Lb=%.6f, Wb=%.6f, ALPHA=%.6f, Ks=%.6f\n",
            10^tmp$par[1], 10^tmp$par[2], 10^tmp$par[3], 10^tmp$par[4]))
      }

      if(all(is.finite(tmp$par))){
        Lb=10^tmp$par[1];Wb=10^tmp$par[2];ALPHA=10^tmp$par[3];Ks=10^tmp$par[4]
        basin_char=c(tmp.area,Lb,X1,Wb,POR)
        gw_hyd=c(ALPHA,BETA,Ks,Kb,Kz)}

      Error=bfs(tmp.q,dys,timestep='day',error_basis='base',basin_char,gw_hyd,flow)
      tmp.bf=bfs_out$Baseflow.L3
      tmp.ov=bfs_out$Baseflow.L3>bfs_out$Qob.L3
      tmp.ov[is.na(tmp.ov)]=FALSE
      tmp.bf[tmp.ov]=bfs_out$Qob.L3[tmp.ov]
      BFF=sum(tmp.bf[!is.na(bfs_out$Qob.L3)])/sum(bfs_out$Qob.L3[!is.na(bfs_out$Qob.L3)])

      # DIAGNOSTIC: After Step 2 final BFS run
      cat(sprintf("\n    === DIAGNOSTIC: STEP 2 - AFTER final BFS run (beta=%.3f) ===\n", BETA))
      cat(sprintf("    Lb=%.6f, X1=%.6f, Wb=%.6f, POR=%.6f\n", Lb, X1, Wb, POR))
      cat(sprintf("    ALPHA=%.6f, BETA=%.6f\n", ALPHA, BETA))
      cat(sprintf("    Ks=%.6f, Kb=%.6f, Kz=%.6f\n", Ks, Kb, Kz))
      cat(sprintf("    Error=%.6f, BFF=%.6f\n", Error, BFF))

      bf_params=rbind(bf_params,data.frame(tmp.site,tmp.area,Lb,X1,Wb,POR,ALPHA,BETA,Ks,Kb,Kz,Qthresh,Rs,Rb1,Rb2,Prec,Frac4Rise,Error,BFF))}
#######################################################################################################
#STEP 3. SELECT BEST PARAMETERS TO MAXIMIZE BASEFLOW AND RE-CALIBRATE SURFACE PARAMETERS
#######################################################################################################
    cat("\n  Step 3: Final calibration with best parameters...\n")
    g=match(max(bf_params$BFF[-1],na.rm=T),bf_params$BFF)
    cat(sprintf("    Selected row %d: BETA=%.3f, BFF=%.6f\n", g, bf_params$BETA[g], bf_params$BFF[g]))
    Lb=bf_params$Lb[g]
    X1=bf_params$X1[g]

    Wb=bf_params$Wb[g]
    basin_char=c(tmp.area,Lb,X1,Wb,POR)

    BETA=bf_params$BETA[g]
    Kb=bf_params$Kb[g]
    Ks=bf_params$Ks[g]
    Kz=bf_params$Kz[g]
    gw_hyd=c(ALPHA,BETA,Ks,Kb,Kz)

    X=c(Lb,Wb,ALPHA,Ks,Kb,Kz)
    LOGX=log(X,10)
    tmp=optim(LOGX,cal_initial,tmp.q=tmp.q,dys=dys,timestep='day',error_basis='base',basin_char=basin_char,gw_hyd=gw_hyd,flow=flow, control=list(maxit=1000,parscale=LOGX,reltol=0.01))
    
    # DIAGNOSTIC: After Step 3 cal_initial
    cat("\n=== DIAGNOSTIC: STEP 3 - AFTER cal_initial ===\n")
    cat(sprintf("Optimization success: %s, iterations: %d, final objective: %.6f\n",
        ifelse(all(is.finite(tmp$par)), "TRUE", "FALSE"), tmp$counts[1], tmp$value))
    if(all(is.finite(tmp$par))) {
      cat(sprintf("Optimized params (log10): %.6f %.6f %.6f %.6f %.6f %.6f\n",
          tmp$par[1], tmp$par[2], tmp$par[3], tmp$par[4], tmp$par[5], tmp$par[6]))
      cat(sprintf("Optimized params (linear): Lb=%.6f, Wb=%.6f, ALPHA=%.6f, Ks=%.6f, Kb=%.6f, Kz=%.6f\n",
          10^tmp$par[1], 10^tmp$par[2], 10^tmp$par[3], 10^tmp$par[4], 10^tmp$par[5], 10^tmp$par[6]))
    }
    
    if(all(is.finite(tmp$par))) {
      Lb=10^tmp$par[1];Wb=10^tmp$par[2];ALPHA=10^tmp$par[3];Ks=10^tmp$par[4];Kb=10^tmp$par[5];Kz=10^tmp$par[6]
      basin_char=c(tmp.area,Lb,X1,Wb,POR)
      gw_hyd=c(ALPHA,BETA,Ks,Kb,Kz)}
#######################################################################################################
#FINAL RUN
#######################################################################################################
    cat("  Running final BFS simulation...\n")
    Error=bfs(tmp.q,dys,timestep='day',error_basis='base',basin_char,gw_hyd,flow)
    tmp.bf=bfs_out$Baseflow.L3
    tmp.ov=bfs_out$Baseflow.L3>bfs_out$Qob.L3
    tmp.ov[is.na(tmp.ov)]=FALSE
    tmp.bf[tmp.ov]=bfs_out$Qob.L3[tmp.ov]
    BFF=sum(tmp.bf[!is.na(bfs_out$Qob.L3)])/sum(bfs_out$Qob.L3[!is.na(bfs_out$Qob.L3)])

    # DIAGNOSTIC: Final parameters (after Step 3)
    cat("\n=== DIAGNOSTIC: FINAL PARAMETERS (after Step 3) ===\n")
    cat(sprintf("Lb=%.6f, X1=%.6f, Wb=%.6f, POR=%.6f\n", Lb, X1, Wb, POR))
    cat(sprintf("ALPHA=%.6f, BETA=%.6f\n", ALPHA, BETA))
    cat(sprintf("Ks=%.6f, Kb=%.6f, Kz=%.6f\n", Ks, Kb, Kz))
    cat(sprintf("Error=%.6f, BFF=%.6f\n", Error, BFF))

    bf_params=data.frame(tmp.site,tmp.area,Lb,X1,Wb,POR,ALPHA,BETA,Ks,Kb,Kz,Qthresh,Rs,Rb1,Rb2,Prec,Frac4Rise,Error,BFF)
    bf_params[,-1]=signif(bf_params[,-1],6)

#GENERATE CREDIBLE INTERVAL TABLE    
    tmp.ci=bf_ci(bfs_out)
    ci_table=tmp.ci[[1]]
#######################################################################################################
#CALCULATE THE COMPONENTS AS FRACTIONS OF STREAMFLOW
    Qmean=mean(bfs_out$Qob.L3,na.rm=T)
    BFF=sum(tmp.bf[!is.na(bfs_out$Qob.L3)])/sum(bfs_out$Qob.L3[!is.na(bfs_out$Qob.L3)])
    SFF=sum(bfs_out$SurfaceFlow.L3[!is.na(bfs_out$Qob.L3)])/sum(bfs_out$Qob.L3[!is.na(bfs_out$Qob.L3)])
    DRF=sum(bfs_out$DirectRunoff.L3[!is.na(bfs_out$Qob.L3)])/sum(bfs_out$Qob.L3[!is.na(bfs_out$Qob.L3)])
    Error=sum(abs(bfs_out$AdjPctEr*bfs_out$Weight),na.rm=T)/sum(bfs_out$Weight,na.rm=T)

    bff=data.frame(tmp.site,Qmean,BFF,SFF,DRF,Error)
    bff[,-1]=signif(bff[,-1],6)

    list(bf_params=bf_params,bff=bff,ci_table=ci_table,bfs_out=bfs_out)
  }}

