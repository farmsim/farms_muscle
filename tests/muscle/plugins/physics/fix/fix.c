/*
 * File: mouse_single_hind.c
 * Date: 23.03.2017
 * Description: Physics plugin to fix the spine of the complete mouse model to the world
 * with a linear slider joint in Y and Z direction to simulate support 
 * Author: Shravan Tata Ramalingasetty
 * Modifications:
 */

#include <ode/ode.h>
#include <plugins/physics.h>


static dBodyID body_base;
static dBodyID body_Reference;

static dJointID joint_Fixed;

void webots_physics_step() {

}

void webots_physics_init()
{
	dWebotsConsolePrintf("Base with support physic plug-in initiated\n");

	//find body in the scene tree
	body_base = dWebotsGetBodyFromDEF("Base");
	if (body_base == NULL)
	{	//we check we found the body
		dWebotsConsolePrintf("ERROR: could not find the body_base in physic plug-in.");
		return;
	}

	dWorldID world = dBodyGetWorld(body_base);
	body_Reference = dBodyCreate (world); // Body to fix to the world frame

	// Create a fixed body attached to the world

	joint_Fixed = dJointCreateFixed(world, 0); // Create a fixed joint with the world

	dJointAttach (joint_Fixed, body_base, 0); // Fix the reference body to the world

	dJointSetFixed (joint_Fixed); // Initialize the joint

}

int webots_physics_collide(dGeomID g1, dGeomID g2) {
	/*
	 * This function needs to be implemented if you want to overide Webots collision detection.
	 * It must return 1 if the collision was handled and 0 otherwise.
	 * Note that contact joints should be added to the contactJointGroup, e.g.
	 *   n = dCollide(g1, g2, MAX_CONTACTS, &contact[0].geom, sizeof(dContact));
	 *   ...
	 *   dJointCreateContact(world, contactJointGroup, &contact[i])
	 *   dJointAttach(contactJoint, body1, body2);
	 *   ...
	 */
	return 0;
}

void webots_physics_cleanup() {
	/*
	 * Here you need to free any memory you allocated in above, close files, etc.
	 * You do not need to free any ODE object, they will be freed by Webots.
	 */
}
