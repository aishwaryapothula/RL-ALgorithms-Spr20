
/*****************************************************************************/
/* File:        Reinf_learn.c                                                */
/* Description: Learning for Cart Pole System                                */
/* Author:                                                                   */
/* Date:                                                                     */
/* Modifications :                                                           */
/*****************************************************************************/

#include <stdio.h>
#include <math.h>
#include <time.h>
#include "Pole_sim.h"
#include "misc.h" 
#include "Reinf_learn.h"


/*****************************************************************************/
/* Multi-dimension decoder according to the discretization in Anderson, 1989 */
/* Input argument indicates the dimension to discretize (0 = X, 1 = Y, 2 = Z)*/
/* and the data structure for the pole-cart system.                          */
/* Return value is the state number (a total of 81 states per dimension).    */
/* Also computes the invert flag used to reduce states space using symmetry, */
/* and what is considered failure and writes it into  fail                   */
/* Input Variables:                                                          */
/*                  axis : the dimension to be encoded                       */
/*                  pole : the data structure of the cart-pole system        */
/* Output Variables:                                                         */
/*                  invert : inversion flag indicating the use of symmetry   */
/*                  fail   : fail flag indicates what is considered failure  */
/* Return Value: Numeric discretized state index for this dimension          */
/*****************************************************************************/
#define STATES  81
#define ACTIONS 2
double =  gamma 0.999;
double Beta = 0.4;
double eps =  0.001;

double Q_values[ACTIONS][STATES];
double Errors[ACTIONS][STATES];
int action;
int prev_action = -1;
int new_state;
int prev_state = -1;
double lamda = 0;

int Decoder3Dn(axis, pole, invert, fail)
int axis;
Polesys *pole;
int *invert;
int *fail;
{
  int pos, vel, ang, avel;
  static double pos_val[4] = {-1.5, -0.5, 0.5, 1.5};
  static double vel_val[4] = {-9999999999.9, -0.5, 0.5, 9999999999.9};
  static double ang_val[7] = {-0.20943951, -0.10471976, -0.017453293, 0.0, 
				        0.017453293, 0.10471976, 0.20943951};
  static double avel_val[4] = {-9999999999.9, -0.87266463, 0.87266463, 
				 9999999999.9};
	
  pos = -1;
  while ((pos < 3) && (pos_val[pos+1] < pole->pos[axis])) 
    ++pos;
  vel = -1;
  while ((vel < 3) && (vel_val[vel+1] < pole->vel[axis])) 
    ++vel;
  if (axis < 2) {
    ang = -1;
    while ((ang < 6) && (ang_val[ang+1] < (pole->theta[1-axis]
					   -(double)(axis)*0.5*M_PI))) 
      ++ang;
    avel = -1;
    while ((avel < 3) && (avel_val[avel+1] < pole->theta_dot[1-axis])) 
      ++avel;
  }
  else {
    ang = -1;
    while ((ang < 6) && (ang_val[ang+1] < MAX(fabs(pole->theta[1]), 
					      fabs(pole->theta[0]-0.5*M_PI)))) 
      ++ang;
    avel = -1;
    while ((avel < 3) && (avel_val[avel+1] < 
			  MAX(SIGN(pole->theta[1])*pole->theta_dot[1],
			      SIGN(pole->theta[0])*pole->theta_dot[0]))) 
      ++avel;
  }
    
  // Sets fail, i.e. if the trial should be considered to have ended based on 
  // this dimension
  *fail = ((pos == -1) || (pos == 3) || (vel == -1) || (vel == 3) || (ang == -1)
 	  || (ang == 6) || (avel == -1) || (avel == 3));

  // Use symmetry to reduce the number of states
  if (!(*fail))
    {
      *invert = 0;
      if (ang > 2)
	     {
		   *invert = 1;
		   ang = 5-ang;
		   pos = 2-pos;
		   vel = 2-vel;
		   avel = 2-avel;
		 }
	  return(pos + 3*vel + 9*ang + 27*avel);
    }
	// Failed situations are not part of the state space
	return(-1);
}

double get_random() { 
  return (double)(rand() / (double)RAND_MAX); 
}



int Sarsa_lamda(reward,explore)
double reward;
int *explore;
{
  int new_action;
  
  if(get_random() < eps){
    new_action = rand()%2;
    (*explore)++;
  }else{
    if (Q_values[0][new_state]  < Q_values[1][new_state])
      new_action = 1;
    else
      new_action = 0;
  }
  double delta = reward + gamma * Q_values[new_action][new_state] - Q_values[prev_action][prev_state];
  Errors[prev_action][prev_state] ++;
  Q_values[prev_action][prev_state] += Beta * delta * Errors[prev_action][prev_state];
  if(prev_action == action && prev_state == new_state){
      Errors[action][new_state]  = gamma * lamda * Errors[prev_action][prev_state] + 1;
  }else{
      Errors[action][new_state]  = gamma * lamda * Errors[prev_action][prev_state];
  }
  return new_action;
}

/*****************************************************************************/
/* Main learning function. Takes the information of the system from   pole   */
/* and the   reset   flag which indicates that this is the first state in a  */
/* new  trial. The action to take in the next time step is written into the  */
/* force    vector and then applied by the simulator to the 3D cart. Also    */
/* returned is the information whether the trial should be ended using the   */
/* fail  flag  and a counter of the number of exploration actions that have  */
/* been taken within this trial by incrementing the  explore  counter every  */
/* an exploration action is taken.                                           */
/* Input Variables:                                                          */
/*                  pole  : the data structure of the cart-pole system       */
/*                  reset : flag indicating that this is the first step in a */
/*                          new trial (i.e. unrelated to the previous state  */
/*                  explore : the number of exploration akitions taken in    */
/*                          this trial prior to this time step               */
/* Output Variables:                                                         */
/*                  force : force vector to be applied to the cart in the    */
/*                          next time step (corresponding to the action taken*/
/*                  fail  : flag indicating whether a new trial should be    */
/*                          started in the next time step                    */
/*                  explore : the number of exploration taken in this trial  */
/*                            including this time step (increase by one if   */
/*                            exploration action was taken)                  */
/*****************************************************************************/
int resetFlag;
void pole_learn(pole, reset, force, fail, explore)
Polesys *pole;
int reset;
double force[3];
int *fail;
int *explore;
{
 
 // Example use of the state discretization decoder
 // Writes state into the state variable and sets the
 // invert flags to keep track of the states that were mapped symmetrically
 // Also sets the correct value to the  fail  flag to restart the simulation
 // trial in the next time step once the pendulum has fallen
  int invert[P_DOF];
  double reward;
  int resetFlag;
  reward = 0.0 - *fail;
  if ( reset )
  {
    *fail = 0;
    *explore = 0;
    prev_state = -1;
    new_state = Decoder3Dn(0, pole, &invert[0], fail);
  }else{   
    prev_state = new_state;
    prev_state = Decoder3Dn(0, pole, &invert[0], fail);
    prev_action = action;
    action = Sarsa_lamda(0.0 - *fail,explore);
    if ( invert[0] )
      force[0] = (double)(2 * action - 1) * -10.0;
    else
      force[0] = (double)(2 * action - 1) * 10.0;
  }
  force[1] = 0.0;
  force[2] = 0.0;
  
}
