f = addfile(ncfile,"r")

out_type="pdf"
out_dir="./fig/spatial/"
system("mkdir -p "+out_dir)

basinstr=tostring(f@basinstr)
models = chartostring(f->model_names)
fullfilename=tostring(f@csvfilename)

; split CSV file extension off
namesplit = get_file_suffix(fullfilename,0)
filename = namesplit@fBase


  lettercount=0
  spapltvarsstr=(/"trackdens","minpres","maxwind","gendens","pacedens","acedens","tcddens", "trackbias","genbias","acebias","pacebias"/)
  spapltvars=(/"fulldens", "fullpres", "fullwind", "fullgen", "fullpace","fullace","fulltcd", "fulltrackbias" , "fullgenbias", "fullacebias", "fullpacebias" /)
  ;spapltmincontour=(/0. ,870. , 0., 0., 0., 0., 0.,-20.,-1.5,-5.,-5./); use this for global
  ;spapltmaxcontour=(/30.,1000.,80., 2., 6., 6., 5., 20., 1.5, 5., 5./)    
  spapltmincontour=(/0. ,870. ,0. ,0. ,0. ,0., 0.,-10.,-1.0,-5.,-5./)  ; use this for NATL
  spapltmaxcontour=(/15.,1000.,80.,1. ,5. ,5., 3., 10., 1.0, 5., 5./)    
  letterstr=(/"a.","b.","c.","d.","e.","f.","g.","h.","i.","j.","k.","l."/)

  ; ==================== do spatial plotting ===========================

  do bb = 0,dimsizes(spapltvarsstr)-1

    wks = gsn_open_wks(out_type,out_dir+"/"+tostring(spapltvarsstr(bb))+"."+basinstr+"_"+filename)


    toPlot=f->$spapltvars(bb)$
    if (bb .eq. 0) then
      dims=dimsizes(toPlot)
      nfiles=dims(0)
    end if
    
    plot = new(nfiles,"graphic")
        
        
    if (spapltvarsstr(bb) .eq. "trackbias" .or. spapltvarsstr(bb) .eq. "genbias" .or. spapltvarsstr(bb) .eq. "acebias" .or. spapltvarsstr(bb) .eq. "pacebias") then
      ;colorMap1="NCV_blu_red"
      ;colorMap1="BlueWhiteOrangeRed"
      ;colorMap1=read_colormap_file("./functions/seaborn_bluetored2.rgb")
      colorMap1=read_colormap_file("NCV_blu_red")
      ; currently a hack to make middle 30 colors = white so that the +/- small contours = white
      colorMap1(111:141,0:2)=1.0
      ncontlev=20
    else
      toPlot = where(toPlot.gt.0.,toPlot,toPlot@_FillValue)
      colorMap1="WhiteBlueGreenYellowRed"
      ncontlev = 10
    end if
    res                       = True     ; plot mods desired
    res@gsnDraw = False
    res@gsnFrame = False
    res@gsnAddCyclic          = False    
    res@cnFillOn              = True     ; turn on color fill
    res@cnFillMode            = "RasterFill"       ; Raster Mode
    res@cnLinesOn             = False    ; turn of contour lines
    res@cnLineLabelsOn  = False

    res@gsnCenterString       = ""
    if (basinstr .eq. "NATL") then
      res@mpMinLatF             = 5.
      res@mpMaxLatF             = 55.
      res@mpMinLonF             = 260.
      res@mpMaxLonF             = 350.
    else
      res@mpMinLatF             = -60.
      res@mpMaxLatF             = 60.
      res@mpMinLonF             = 0.
      res@mpMaxLonF             = 360.
    end if
    res@mpCenterLonF          = (res@mpMinLonF + res@mpMaxLonF)/2
    res@cnLevelSelectionMode  = "ManualLevels"
    res@cnMinLevelValF        = spapltmincontour(bb)
    res@cnMaxLevelValF        = spapltmaxcontour(bb)
    res@cnLevelSpacingF       = (res@cnMaxLevelValF - res@cnMinLevelValF) / tofloat(ncontlev)
    res@cnFillPalette=colorMap1

    res@lbLabelStride = 2
    res@lbLabelBarOn        = False           ; turn off individual cb's

    do zz = 0,nfiles-1
      plot(zz) = gsn_csm_contour_map_ce(wks,toPlot(zz,:,:),res)
    end do

    resP=True
    ;resP@txString           = "Annual TC track density, all storms"
    resP@amJust           = "TopLeft"
    resP@gsnPanelFigureStringsFontHeightF = 0.015
    ;resP@gsnPanelFigureStrings="valid_strs"

    resP@gsnPanelLabelBar    = True                ; add common colorbar
    resP@lbLabelFontHeightF  = 0.012               ; make labels smaller

    if (nfiles .le. 4) then
      gsn_panel(wks,plot,(/4,1/),resP)
    else if (nfiles .gt. 4 .and. nfiles .le. 8) then
      gsn_panel(wks,plot,(/4,2/),resP)
    else
      gsn_panel(wks,plot,(/4,3/),resP)
    end if
    end if

    delete(res)
    delete(colorMap1)
    delete(plot)
    delete(toPlot)
    delete(wks)
  end do