/*
console.log('Loaded!');

// Change the text of the main-text div

var element = document.getElementById('main-text');

element.innerHTML = 'New value';

// Move the image

var img = document.getElementById('madi');

var marginLeft = 0;
function moveRight () 
{
    marginLeft = marginLeft +1;
    img.style.marginLeft = marginLeft+'px';
}
img.onclick = function ()
{
    var interval = setInterval(moveRight, 50);
    //img.style.marginLeft = '100px';
};
*/

// Counter code

var button = document.getElementById('counter');
var counter = 0;
button.onclick = function ()
{
  // Make a request to the counter endpoint
  
  // Capture the response and store it in a variable
  
  // Render a variable in the correct span
  counter = counter+1;
  var span = document.getElementById('count');
  span.innerHTML = counter.toString();
};