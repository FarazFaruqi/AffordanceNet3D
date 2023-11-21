# Affordance Control for 3D Reconstruction


## TODO

### General 
- [ ] Find and implement new test cases other than chair and table
- [ ] Find other objective functions (fabrication, stability, rigid motion?)
### FEA 
- Push for higher resolution:
  - [ ] use sparse matrices for K matrix and force/displacement matrices, find efficient sparse matrix inversion library
  - [x] implement neural implicit filed with fourier projection in volume renderer
  - [ ] FEA small mesh and reuse Neural implicit field to get higher resolution models
  - [ ] implement dynamics simulation similar to [this paper](https://openaccess.thecvf.com/content/CVPR2022/html/Hong_Fixing_Malfunctional_Objects_With_Learned_Physical_Simulation_and_Functional_Prediction_CVPR_2022_paper.html)
### Affordance
- [ ] Automatic Boundary Conditions from text input
  - [ ] Integrate 3D AffordanceNet and test on different chair and table models we have to see how robust
  - [ ] sampling method to turn our 3d volume into point cloud to be used in 3d affordance net
  - [ ] Transform affordancenet output into boundary conditions (forces and fixed regions) to FEA model
  - [ ] figure out text to affordance workflow in terms of UI
### Diffusion model
- [ ] choose most appropriate model to integrate
- [ ] Link to text to 2D images/views diffusion model
  - [ ] make sure it works without FEA constraint
  - [ ] Integrate with FEA


## Using Geometry for Objective Function
- [ ] For the chair example, find the goal set of affordances
- [ ] For these specific set of affordances, find models from 3DAffordanceNet that have them
- [ ] Segment these models, and find the affordance related segments
- [ ] During the generation process, optimize different sections to imitate these segments 

### Experiment 1: All segments come from a single model
For the chair example, segment an existing chair, and make the corresponding parts get losses from corresponding parts. 
This would be similar to comparing it to an existing complete chair, but the losses would be coming from segments instead of the global model. 

### Experiment 2: Similarities between segments having the same affordance 
- [ ] For each affordance, take a subset of 3D models from 3DAffordanceNet, and segment them. 
- [ ] Find the specific segments for each affordance label, and categorize them accordingly. 
- [ ] Get similarity features between these segments. 
##### Question: Are segments having the same affordance label, similar to each other?  

### Experiment 2: Segments coming from different models
For the chair example, get parts corresponding parts get losses from corresponding parts. 
This would be similar to comparing it to an existing complete chair, but the losses would be coming from segments of different models instead of the same model. 

---