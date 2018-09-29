// const second = () => {
//     setTimeout(() => {
//         console.log('Async Hey There');
//     }, 2000);
// };
//
// const first = () => {
//     console.log('Hey there');
//     second();
//     console.log('The end');
// };
//
// first();

// callback hell
// getRecipe = () => {
//     setTimeout(() => {
//         const recipeID = [523, 883, 432, 974];
//
//         console.log(recipeID);
//
//         setTimeout((id) => {
//             const recipe = {title: 'Fresh tomato pasta', publisher: 'Jesse'};
//             console.log(`${id}: ${recipe.title}`);
//
//             setTimeout((publisher) => {
//                 const recipe = {title: 'Italian pizza', publisher: 'Jesse'};
//
//                 console.log(recipe);
//             }, 1500, recipe.publisher)
//         }, 1500, recipeID[2])
//     }, 1500)
// };
//
// getRecipe();

// const getIDs = new Promise((resolve, reject) => {
//     setTimeout(() => {
//         resolve([523, 883, 432, 974]);
//     }, 1500);
// });
//
// const getRecipe = recID => {
//     return new Promise((resolve, reject) => {
//         setTimeout(ID => {
//             const recipe = {title: 'Fresh tomato pasta', publisher: 'Jesse'};
//             resolve(`${ID}: ${recipe.title}`);
//         }, 1500, recID)
//     })
// };
//
// const getRelated = publisher => {
//     return new Promise((resolve, reject) => {
//         setTimeout(pub => {
//             const recipe = {title: 'Italian pizza', publisher: 'Jesse'};
//             resolve(`${pub}: ${recipe.title}`);
//         }, 1500, publisher)
//     })
// };
//
// getIDs.then((IDs) => {
//     console.log(IDs);
//     return getRecipe(IDs[2]);
// }).then(recipe => {
//     console.log(recipe);
//     return getRelated('Jesse');
// }).then(recipe => {
//     console.log(recipe);
// }).catch((error) => {
//     console.log(error);
// });

// async function getRecipesAW() {
//     try {
//         const IDs = await getIDs;
//         console.log(IDs);
//         const recipe = await getRecipe(IDs[2]);
//         console.log(recipe);
//         const related = await getRelated('Jesse');
//         console.log(related);
//
//         return recipe;
//     } catch(error) {
//         throw Error(error);
//     }
// };
//
// getRecipesAW().then(result => console.log(result));

// fetch('https://crossorigin.me/https://www.metaweather.com/api/location/2487956/')
// function getWeather(woeid) {
//     fetch(`https://cors-anywhere.herokuapp.com/https://www.metaweather.com/api/location/${woeid}/`)
//         .then(result => {
//             // console.log(result);
//             return result.json();
//         })
//         .then(result => {
//             // console.log(result);
//             const today = result.consolidated_weather[0];
//             console.log(`Temperature in ${result.title} stay between ${today.min_temp} and ${today.max_temp}`)
//         })
//         .catch(e => console.log(`Error: ${e}`));
// }
//
// getWeather(2487956);
// getWeather(44418);

async function getWeather(woeid) {
    try {
        const res = await fetch(`https://cors-anywhere.herokuapp.com/https://www.metaweather.com/api/location/${woeid}/`);
        const data = await res.json();
        const tomorrow = data.consolidated_weather[1];
        console.log(`Temperature in ${data.title} stay between ${tomorrow.min_temp} and ${tomorrow.max_temp}`)

        return data;
    } catch (e) {
        throw Error(e);
    }
}

getWeather(2487956).then(res => console.log(res));
getWeather(44418).then(res => console.log(res));